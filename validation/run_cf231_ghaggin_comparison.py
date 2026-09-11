"""Compare the analytic InEKF with the original ghaggin C++ propagation.

The original reference implementation is compiled without source changes.
Because it fixes gravity to 9.81 m/s² and does not expose a bias setter, every
method in this benchmark uses gravity 9.81 and receives the same fixed-bias
correction.  Run-5 GT is used only for initialization and evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.cf231_loader import load_cf231_run, synchronize_to_imu
from validation.run_cf231_learned_velocity_leave5 import learned_propagation
from validation.run_cf231_sensor_frame_leave5 import (
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
    gravity_aligned_imu_rotation,
    propagate,
)

REFERENCE_GRAVITY = 9.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "cf231_leave_one_out" / "csv",
    )
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--bias-runs", default="3,9,10")
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--learned-results",
        type=Path,
        default=ROOT
        / "validation"
        / "results"
        / "cf231_leave5_learned_velocity",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GHAGGIN_INEKF_ROOT",
                str(
                    ROOT
                    / "third_party"
                    / "ghaggin-invariant-ekf"
                    / "inekf"
                ),
            )
        ),
        help="Path to the upstream repository's inekf directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation"
        / "results"
        / "cf231_run5_ghaggin_comparison",
    )
    return parser.parse_args()


def parse_run_ids(text: str) -> list[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def compile_reference(reference_root: Path, executable: Path) -> None:
    required = (
        reference_root / "include" / "IEKF.hpp",
        reference_root / "include" / "utils.hpp",
        reference_root / "src" / "IEKF.cpp",
        reference_root / "src" / "utils.cpp",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing ghaggin reference files: " + ", ".join(missing)
        )
    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        f"-I{reference_root / 'include'}",
        str(ROOT / "validation" / "reference" / "ghaggin_cf231_runner.cpp"),
        str(reference_root / "src" / "IEKF.cpp"),
        str(reference_root / "src" / "utils.cpp"),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True)


def write_reference_input(
    path: Path,
    run,
    corrected_imu: np.ndarray,
    initial_rotation: np.ndarray,
) -> None:
    initial_state = np.r_[
        initial_rotation.ravel(),
        run.velocity[0],
        run.position[0],
    ]
    repeated_state = np.repeat(
        initial_state.reshape(1, -1),
        run.time.size,
        axis=0,
    )
    values = np.column_stack((run.time, corrected_imu, repeated_state))
    header = (
        "time,ax,ay,az,gx,gy,gz,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,"
        "vx,vy,vz,px,py,pz"
    )
    np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.17g")


def load_reference_output(
    path: Path,
    run,
    initial_imu_rotation: np.ndarray,
) -> Trajectory:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    if values.shape != (run.time.size, 16):
        raise ValueError(
            f"Unexpected reference output shape {values.shape}; "
            f"expected {(run.time.size, 16)}."
        )
    if not np.allclose(values[:, 0], run.time, atol=1.0e-12, rtol=0.0):
        raise ValueError("Reference output timestamps do not match the IMU input.")
    imu_rotation = values[:, 1:10].reshape(-1, 3, 3)
    initial_body_rotation = run.rotation[0].as_matrix()
    body_from_imu = initial_body_rotation.T @ initial_imu_rotation
    body_rotation = np.einsum(
        "nij,jk->nik",
        imu_rotation,
        body_from_imu.T,
    )
    zeros = np.zeros((run.time.size, 3))
    return Trajectory(
        time=run.time.copy(),
        imu_rotation=imu_rotation,
        body_rotation=body_rotation,
        velocity=values[:, 10:13],
        position=values[:, 13:16],
        position_std=zeros.copy(),
        attitude_std_deg=zeros.copy(),
    )


def orientation_difference_deg(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", left, right)
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def rpy_and_error(
    trajectory: Trajectory,
    gt_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rpy = Rotation.from_matrix(trajectory.body_rotation).as_euler(
        "xyz",
        degrees=True,
    )
    gt_rpy = Rotation.from_matrix(gt_rotation).as_euler("xyz", degrees=True)
    error = (rpy - gt_rpy + 180.0) % 360.0 - 180.0
    return rpy, error


def plot_implementation_difference(
    output: Path,
    time: np.ndarray,
    rotation_difference: np.ndarray,
    velocity_difference: np.ndarray,
    position_difference: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    values = (
        (rotation_difference, "SO(3) difference [deg]"),
        (velocity_difference, "velocity difference [m/s]"),
        (position_difference, "position difference [m]"),
    )
    for axis, (difference, label) in zip(axes, values):
        axis.semilogy(time, np.maximum(difference, 1.0e-16), color="tab:purple")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_title(
        "Analytic InEKF minus original ghaggin C++ propagation"
    )
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rpy(
    output: Path,
    run,
    trajectories: dict[str, Trajectory],
) -> None:
    gt_rotation = run.rotation.as_matrix()
    gt_rpy = np.rad2deg(np.unwrap(run.rotation.as_euler("xyz"), axis=0))
    colors = {
        "ghaggin C++": "tab:orange",
        "analytic InEKF": "tab:green",
        "learned velocity + InEKF": "tab:blue",
    }
    styles = {
        "ghaggin C++": "--",
        "analytic InEKF": ":",
        "learned velocity + InEKF": "-",
    }
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 11),
        sharex="col",
        constrained_layout=True,
    )
    labels = ("roll", "pitch", "yaw")
    for axis_index, label in enumerate(labels):
        axes[axis_index, 0].plot(
            run.time,
            gt_rpy[:, axis_index],
            "k",
            lw=1.2,
            label="GT",
        )
        for name, trajectory in trajectories.items():
            rpy, error = rpy_and_error(trajectory, gt_rotation)
            unwrapped_rpy = np.rad2deg(
                np.unwrap(
                    Rotation.from_matrix(trajectory.body_rotation).as_euler(
                        "xyz"
                    ),
                    axis=0,
                )
            )
            axes[axis_index, 0].plot(
                run.time,
                unwrapped_rpy[:, axis_index],
                color=colors[name],
                ls=styles[name],
                lw=1.0,
                label=name,
            )
            rmse = float(
                np.sqrt(np.mean(np.square(error[:, axis_index])))
            )
            axes[axis_index, 1].plot(
                run.time,
                error[:, axis_index],
                color=colors[name],
                ls=styles[name],
                lw=1.0,
                label=f"{name}: {rmse:.2f}°",
            )
        axes[axis_index, 0].set_ylabel(f"{label} [deg]")
        axes[axis_index, 1].set_ylabel(f"{label} error [deg]")
        axes[axis_index, 0].grid(True, alpha=0.3)
        axes[axis_index, 1].grid(True, alpha=0.3)
        axes[axis_index, 1].axhline(0.0, color="k", lw=0.6, alpha=0.5)
        axes[axis_index, 1].legend(loc="best", fontsize=8)
    axes[0, 0].set_title("RPY estimates")
    axes[0, 1].set_title("Wrapped error relative to GT")
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_gt_errors(
    output: Path,
    run,
    trajectories: dict[str, Trajectory],
) -> None:
    colors = {
        "ghaggin C++": "tab:orange",
        "analytic InEKF": "tab:green",
        "learned velocity + InEKF": "tab:blue",
    }
    styles = {
        "ghaggin C++": "--",
        "analytic InEKF": ":",
        "learned velocity + InEKF": "-",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, trajectory in trajectories.items():
        position_error = np.linalg.norm(
            trajectory.position - run.position,
            axis=1,
        )
        attitude_error = orientation_difference_deg(
            run.rotation.as_matrix(),
            trajectory.body_rotation,
        )
        axes[0].semilogy(
            run.time,
            np.maximum(position_error, 1.0e-6),
            color=colors[name],
            ls=styles[name],
            label=name,
        )
        axes[1].plot(
            run.time,
            attitude_error,
            color=colors[name],
            ls=styles[name],
            label=name,
        )
    axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    axes[0].set_title("Error relative to run-5 GT")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_trajectory(
    output: Path,
    run,
    trajectories: dict[str, Trajectory],
) -> None:
    colors = {
        "ghaggin C++": "tab:orange",
        "analytic InEKF": "tab:green",
        "learned velocity + InEKF": "tab:blue",
    }
    styles = {
        "ghaggin C++": "--",
        "analytic InEKF": ":",
        "learned velocity + InEKF": "-",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    twenty_seconds = int(np.searchsorted(run.time, 20.0, side="right"))
    for axis, cutoff, title in (
        (axes[0], twenty_seconds, "First 20 seconds"),
        (axes[1], run.time.size, "Full trajectory near the GT range"),
    ):
        axis.plot(
            run.position[:cutoff, 0],
            run.position[:cutoff, 1],
            "k",
            lw=1.4,
            label="GT",
        )
        for name, trajectory in trajectories.items():
            axis.plot(
                trajectory.position[:cutoff, 0],
                trajectory.position[:cutoff, 1],
                color=colors[name],
                ls=styles[name],
                lw=1.1,
                label=name,
            )
        axis.scatter(
            run.position[0, 0],
            run.position[0, 1],
            c="black",
            s=35,
            zorder=5,
        )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend()

    margin_x = max(1.0, 0.55 * np.ptp(run.position[:, 0]))
    margin_y = max(0.8, 0.55 * np.ptp(run.position[:, 1]))
    axes[1].set_xlim(
        np.min(run.position[:, 0]) - margin_x,
        np.max(run.position[:, 0]) + margin_x,
    )
    axes[1].set_ylim(
        np.min(run.position[:, 1]) - margin_y,
        np.max(run.position[:, 1]) + margin_y,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def upstream_commit(reference_root: Path) -> str:
    repository = reference_root.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    learned_results = args.learned_results.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("Loading run 5 and estimating one fixed bias from other runs...")
    run = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    calibration = estimate_sensor_calibration(
        dataset,
        parse_run_ids(args.bias_runs),
        args.static_seconds,
    )
    initial_rotation, initial_alignment = gravity_aligned_imu_rotation(
        run,
        calibration.accel_bias,
        args.static_seconds,
    )
    corrected_imu = run.imu.copy()
    corrected_imu[:, 0:3] -= calibration.accel_bias
    corrected_imu[:, 3:6] -= calibration.gyro_bias

    reference_input = output / "reference_input.csv"
    reference_output = output / "reference_output.csv"
    reference_executable = output / "build" / "ghaggin_cf231_runner"
    write_reference_input(
        reference_input,
        run,
        corrected_imu,
        initial_rotation,
    )
    print("Compiling and running the unmodified ghaggin C++ propagation...")
    compile_reference(reference_root, reference_executable)
    subprocess.run(
        [
            str(reference_executable),
            str(reference_input),
            str(reference_output),
        ],
        check=True,
    )
    reference = load_reference_output(
        reference_output,
        run,
        initial_rotation,
    )

    print("Running the analytic InEKF with exactly the same gravity and bias...")
    analytic = propagate(
        run,
        calibration,
        initial_rotation,
        gravity_magnitude=REFERENCE_GRAVITY,
    )

    print("Replaying the held-out learned velocity measurements at gravity 9.81...")
    learned_archive = np.load(learned_results / "trajectories.npz")
    learned_summary = json.loads(
        (learned_results / "summary.json").read_text(encoding="utf-8")
    )
    velocity_variance = np.asarray(
        learned_summary["cross_validation"][
            "pooled_velocity_variance_m2_s2"
        ],
        dtype=float,
    )
    covariance_inflation = float(
        learned_summary["configuration"]["measurement_covariance_inflation"]
    )
    learned, _, _ = learned_propagation(
        run,
        calibration,
        initial_rotation,
        learned_archive["update_indices"],
        learned_archive["predicted_heading_velocity"],
        velocity_variance,
        covariance_inflation,
        gravity_magnitude=REFERENCE_GRAVITY,
    )

    implementation_rotation = orientation_difference_deg(
        reference.imu_rotation,
        analytic.imu_rotation,
    )
    implementation_velocity = np.linalg.norm(
        reference.velocity - analytic.velocity,
        axis=1,
    )
    implementation_position = np.linalg.norm(
        reference.position - analytic.position,
        axis=1,
    )

    trajectories = {
        "ghaggin C++": reference,
        "analytic InEKF": analytic,
        "learned velocity + InEKF": learned,
    }
    metrics: dict[str, dict[str, object]] = {}
    for name, trajectory in trajectories.items():
        metric, _, _ = calculate_metrics(trajectory, run)
        metrics[name] = metric

    agreement = {
        "max_so3_difference_deg": float(np.max(implementation_rotation)),
        "max_velocity_difference_m_s": float(np.max(implementation_velocity)),
        "max_position_difference_m": float(np.max(implementation_position)),
        "final_so3_difference_deg": float(implementation_rotation[-1]),
        "final_velocity_difference_m_s": float(implementation_velocity[-1]),
        "final_position_difference_m": float(implementation_position[-1]),
    }
    thresholds = {
        "so3_difference_deg": 1.0e-4,
        "velocity_difference_m_s": 1.0e-4,
        "position_difference_m": 1.0e-2,
    }
    agreement["passes_target"] = bool(
        agreement["max_so3_difference_deg"]
        <= thresholds["so3_difference_deg"]
        and agreement["max_velocity_difference_m_s"]
        <= thresholds["velocity_difference_m_s"]
        and agreement["max_position_difference_m"]
        <= thresholds["position_difference_m"]
    )

    payload = {
        "configuration": {
            "dataset": str(dataset),
            "test_run": args.test_run,
            "bias_runs": parse_run_ids(args.bias_runs),
            "reference_root": str(reference_root),
            "reference_commit": upstream_commit(reference_root),
            "reference_gravity_m_s2": REFERENCE_GRAVITY,
            "reference_source_modified": False,
            "reference_bias_handling": (
                "the fixed bias is subtracted before addImu because the "
                "upstream class has no public bias setter"
            ),
            "same_initial_rotation_velocity_position": True,
            "uses_measurement_updates_for_fixed_methods": False,
            "run5_gt_use": "one initial state and post-run evaluation only",
        },
        "fixed_bias": {
            "gyro_rad_s": calibration.gyro_bias.tolist(),
            "accel_m_s2": calibration.accel_bias.tolist(),
        },
        "initial_alignment": initial_alignment,
        "implementation_agreement": agreement,
        "acceptance_thresholds": thresholds,
        "gt_metrics": metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    plot_implementation_difference(
        figures / "01_implementation_difference.png",
        run.time,
        implementation_rotation,
        implementation_velocity,
        implementation_position,
    )
    plot_rpy(
        figures / "02_rpy_comparison.png",
        run,
        trajectories,
    )
    plot_gt_errors(
        figures / "03_gt_error_comparison.png",
        run,
        trajectories,
    )
    plot_trajectory(
        figures / "04_trajectory_comparison.png",
        run,
        trajectories,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
