"""Continuous EuRoC IMU-only orientation validation for the analytic InEKF.

One fixed bias and one initial GT navigation state are used. After the
evaluation start there are no GT resets, position updates, pseudo-velocity
updates, or other measurements. Ground truth is used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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

from filters.Hoon_invariant_kalman_analytic_15D import (
    HoonInvariantKalmanAnalytic15D,
)
from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import exp_so3, gamma2_so3, left_jacobian_so3
from utils.imu_bias import estimate_imu_bias_from_gt
from validation.euroc_loader import EurocData, load_euroc_sequence

GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass(frozen=True)
class FixedBias:
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    attitude_std_deg: np.ndarray | None = None


def make_filter() -> HoonInvariantKalmanAnalytic15D:
    return HoonInvariantKalmanAnalytic15D(
        mode="imu_only",
        motion_config={
            "gravity": GRAVITY,
            "jacobian_mode": "analytic",
            "translation_input_frame": "body",
            "translation_input_type": "acceleration",
            "rotation_input_type": "rate",
            "update_biases": False,
        },
    )


def choose_fixed_bias(
    data: EurocData,
    calibration_seconds: float,
    source: str,
) -> FixedBias:
    calibration_mask = data.time <= calibration_seconds
    if np.count_nonzero(calibration_mask) < 2:
        raise ValueError("Bias calibration interval contains fewer than two samples.")
    if source == "euroc_gt":
        return FixedBias(
            gyro=np.mean(data.gyro_bias[calibration_mask], axis=0),
            accel=np.mean(data.accel_bias[calibration_mask], axis=0),
        )
    estimate = estimate_imu_bias_from_gt(
        data.time,
        data.imu,
        data.rotation.as_matrix(),
        data.velocity,
        gravity=GRAVITY,
        calibration_range=(0.0, calibration_seconds),
        statistic="median",
    )
    return FixedBias(gyro=estimate.gyro, accel=estimate.accel)


def run_inekf(
    data: EurocData,
    start: int,
    end: int,
    bias: FixedBias,
) -> Trajectory:
    """Run the actual analytic InEKF continuously, without any update."""
    estimator = make_filter()
    estimator.Rot = data.rotation[start].as_matrix().copy()
    estimator.v = data.velocity[start].copy()
    estimator.p = data.position[start].copy()
    estimator.set_fixed_imu_bias(bias.gyro, bias.accel)
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)

    rotations = [estimator.Rot.copy()]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    attitude_std = [np.rad2deg(np.sqrt(np.clip(np.diag(estimator.P)[:3], 0.0, None)))]
    for index in range(start + 1, end + 1):
        dt = float(data.time[index] - data.time[index - 1])
        # ghaggin/invariant-ekf propagates with the IMU sample delivered at
        # the current timestamp and the elapsed time since the prior sample.
        estimator.predict(data.imu[index], dt)
        rotations.append(estimator.Rot.copy())
        velocities.append(estimator.v.copy())
        positions.append(estimator.p.copy())
        attitude_std.append(
            np.rad2deg(np.sqrt(np.clip(np.diag(estimator.P)[:3], 0.0, None)))
        )
    return Trajectory(
        time=data.time[start : end + 1] - data.time[start],
        rotation=np.asarray(rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
        attitude_std_deg=np.asarray(attitude_std),
    )


def run_reference_propagation(
    data: EurocData,
    start: int,
    end: int,
    bias: FixedBias,
) -> Trajectory:
    """Python translation of ghaggin's LIEKF mean propagation."""
    rotation = data.rotation[start].as_matrix().copy()
    velocity = data.velocity[start].copy()
    position = data.position[start].copy()
    rotations = [rotation.copy()]
    velocities = [velocity.copy()]
    positions = [position.copy()]
    for index in range(start + 1, end + 1):
        dt = float(data.time[index] - data.time[index - 1])
        imu = data.imu[index]
        omega = imu[3:6] - bias.gyro
        accel = imu[0:3] - bias.accel
        phi = omega * dt
        position = (
            position
            + velocity * dt
            + rotation @ gamma2_so3(phi) @ accel * dt * dt
            + 0.5 * GRAVITY * dt * dt
        )
        velocity = (
            velocity
            + rotation @ left_jacobian_so3(phi) @ accel * dt
            + GRAVITY * dt
        )
        rotation = rotation @ exp_so3(omega * dt)
        rotations.append(rotation.copy())
        velocities.append(velocity.copy())
        positions.append(position.copy())
    return Trajectory(
        time=data.time[start : end + 1] - data.time[start],
        rotation=np.asarray(rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
    )


def wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def orientation_errors(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    estimated_rpy = Rotation.from_matrix(estimated).as_euler("xyz", degrees=True)
    gt_rpy = Rotation.from_matrix(ground_truth).as_euler("xyz", degrees=True)
    rpy_error = wrap_degrees(estimated_rpy - gt_rpy)
    relative = np.einsum("nji,njk->nik", ground_truth, estimated)
    geodesic = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    return estimated_rpy, gt_rpy, rpy_error, geodesic


def metric_block(rpy_error: np.ndarray, geodesic: np.ndarray) -> dict:
    labels = ["roll", "pitch", "yaw"]
    result: dict[str, float | dict] = {}
    for axis, label in enumerate(labels):
        values = np.abs(rpy_error[:, axis])
        result[label] = {
            "rmse_deg": float(np.sqrt(np.mean(rpy_error[:, axis] ** 2))),
            "median_abs_deg": float(np.median(values)),
            "p95_abs_deg": float(np.percentile(values, 95)),
            "max_abs_deg": float(np.max(values)),
            "final_abs_deg": float(values[-1]),
        }
    result["geodesic"] = {
        "rmse_deg": float(np.sqrt(np.mean(geodesic**2))),
        "median_deg": float(np.median(geodesic)),
        "p95_deg": float(np.percentile(geodesic, 95)),
        "max_deg": float(np.max(geodesic)),
        "final_deg": float(geodesic[-1]),
    }
    return result


def summarize_horizons(
    time: np.ndarray,
    rpy_error: np.ndarray,
    geodesic: np.ndarray,
) -> dict[str, dict]:
    horizons = [10.0, 30.0, 60.0, 120.0, float(time[-1])]
    summaries: dict[str, dict] = {}
    for horizon in horizons:
        mask = time <= min(horizon, float(time[-1])) + 1e-9
        key = "full" if horizon == float(time[-1]) else f"{horizon:g}s"
        summaries[key] = metric_block(rpy_error[mask], geodesic[mask])
    return summaries


def plot_rpy(
    time: np.ndarray,
    gt_rpy: np.ndarray,
    fixed_rpy: np.ndarray,
    reference_rpy: np.ndarray,
    output: Path,
) -> None:
    labels = ["roll", "pitch", "yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for axis, label in enumerate(labels):
        axes[axis].plot(time, gt_rpy[:, axis], "k", lw=1.2, label="GT")
        axes[axis].plot(time, fixed_rpy[:, axis], color="tab:blue", lw=0.9, label="analytic InEKF")
        axes[axis].plot(
            time,
            reference_rpy[:, axis],
            color="tab:orange",
            lw=0.8,
            ls="--",
            label="ghaggin LIEKF propagation",
        )
        axes[axis].set_ylabel(f"{label} [deg]")
        axes[axis].grid(True)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("time since initial state [s]")
    fig.suptitle("Continuous IMU-only orientation: one initial state, one fixed bias")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rpy_errors(
    time: np.ndarray,
    fixed_error: np.ndarray,
    zero_error: np.ndarray,
    output: Path,
) -> None:
    labels = ["roll", "pitch", "yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for axis, label in enumerate(labels):
        axes[axis].plot(time, zero_error[:, axis], color="0.65", lw=0.8, label="zero bias")
        axes[axis].plot(time, fixed_error[:, axis], color="tab:blue", lw=0.9, label="one fixed bias")
        axes[axis].axhline(0.0, color="black", lw=0.6)
        axes[axis].set_ylabel(f"{label} error [deg]")
        axes[axis].grid(True)
    axes[0].legend()
    axes[-1].set_xlabel("time since initial state [s]")
    fig.suptitle("Wrapped Euler-angle error (estimate - GT)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_geodesic_error(
    time: np.ndarray,
    fixed_error: np.ndarray,
    zero_error: np.ndarray,
    reference_error: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(time, zero_error, color="0.65", lw=0.8, label="zero bias")
    ax.plot(time, fixed_error, color="tab:blue", lw=1.0, label="analytic InEKF")
    ax.plot(
        time,
        reference_error,
        color="tab:orange",
        lw=0.8,
        ls="--",
        label="ghaggin LIEKF propagation",
    )
    ax.set_xlabel("time since initial state [s]")
    ax.set_ylabel("SO(3) geodesic error [deg]")
    ax.set_title("Continuous IMU-only attitude error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_trajectory_diagnostic(
    time: np.ndarray,
    gt_position: np.ndarray,
    fixed: Trajectory,
    reference: Trajectory,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, horizon in zip(axes, [10.0, float(time[-1])]):
        mask = time <= horizon + 1e-9
        ax.plot(gt_position[mask, 0], gt_position[mask, 1], "k--", lw=1.5, label="GT")
        ax.plot(fixed.position[mask, 0], fixed.position[mask, 1], color="tab:blue", label="analytic InEKF")
        ax.plot(
            reference.position[mask, 0],
            reference.position[mask, 1],
            color="tab:orange",
            ls="--",
            label="ghaggin LIEKF propagation",
        )
        ax.set_title(f"First {horizon:g} s" if horizon < time[-1] else "Full interval")
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.axis("equal")
        ax.grid(True)
    axes[0].legend()
    fig.suptitle("Trajectory diagnostic (not the primary IMU-only metric)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    data: EurocData,
    start: int,
    bias: FixedBias,
    bias_source: str,
    summaries: dict[str, dict],
    reference_difference_deg: float,
    position_error_10s: float,
    position_error_final: float,
) -> None:
    full = summaries["full"]
    lines = [
        "# Continuous EuRoC IMU-only InEKF validation",
        "",
        f"- Sequence: `{data.source_dir}`",
        f"- Bias calibration interval: 0–{data.time[start]:.3f} s",
        f"- Evaluation interval: {data.time[start]:.3f}–{data.time[-1]:.3f} s",
        "- GT is used once to initialize R, v, p at the evaluation start.",
        "- No position, velocity, orientation, GPS, or pseudo-velocity update is used afterward.",
        f"- Fixed bias source: `{bias_source}`",
        f"- Fixed gyro bias [rad/s]: `{np.array2string(bias.gyro, precision=8)}`",
        f"- Fixed accel bias [m/s²]: `{np.array2string(bias.accel, precision=8)}`",
        "",
        "## Full-interval orientation error",
        "",
        "| Component | RMSE [deg] | Median abs [deg] | 95% abs [deg] | Final abs [deg] |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["roll", "pitch", "yaw"]:
        metric = full[label]
        lines.append(
            f"| {label} | {metric['rmse_deg']:.6g} | "
            f"{metric['median_abs_deg']:.6g} | {metric['p95_abs_deg']:.6g} | "
            f"{metric['final_abs_deg']:.6g} |"
        )
    geo = full["geodesic"]
    lines += [
        "",
        f"- SO(3) geodesic RMSE: {geo['rmse_deg']:.6g} deg",
        f"- SO(3) geodesic 95%: {geo['p95_deg']:.6g} deg",
        f"- Maximum difference from ghaggin LIEKF mean propagation: {reference_difference_deg:.6g} deg",
        "",
        "Roll/yaw Euler components are sensitive because this sequence operates",
        "near a high-pitch attitude. The SO(3) geodesic error is therefore the",
        "coordinate-free primary attitude metric; RPY errors are retained as",
        "requested diagnostics.",
        "",
        "## Trajectory drift diagnostic",
        "",
        f"- Position error at 10 s: {position_error_10s:.6g} m",
        f"- Final position error: {position_error_final:.6g} m",
        "",
        "Trajectory is shown only as a drift diagnostic. Without an external",
        "measurement, the Kalman covariance cannot correct the nominal IMU-only",
        "trajectory, so the primary result is the orientation error above.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=ROOT / "data" / "euroc" / "V1_01_easy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "results" / "euroc_v1_01_continuous",
    )
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument(
        "--bias-source",
        choices=["euroc_gt", "gt_residual"],
        default="euroc_gt",
        help="Both choices produce one constant bias from the calibration interval.",
    )
    parser.add_argument(
        "--evaluation-seconds",
        type=float,
        default=None,
        help="Optional evaluation length after calibration; default uses all data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    data = load_euroc_sequence(args.sequence)
    start = int(np.searchsorted(data.time, args.calibration_seconds, side="left"))
    end_time = float(data.time[-1])
    if args.evaluation_seconds is not None:
        end_time = min(end_time, float(data.time[start] + args.evaluation_seconds))
    end = int(np.searchsorted(data.time, end_time, side="right") - 1)
    if not 0 < start < end < len(data.time):
        raise ValueError("Invalid calibration/evaluation interval.")

    fixed_bias = choose_fixed_bias(data, args.calibration_seconds, args.bias_source)
    zero_bias = FixedBias(np.zeros(3), np.zeros(3))
    print("Running continuous analytic InEKF with one fixed bias...")
    fixed = run_inekf(data, start, end, fixed_bias)
    print("Running zero-bias baseline...")
    zero = run_inekf(data, start, end, zero_bias)
    print("Running ghaggin LIEKF mean-propagation comparison...")
    reference = run_reference_propagation(data, start, end, fixed_bias)

    gt_rotation = data.rotation.as_matrix()[start : end + 1]
    gt_position = data.position[start : end + 1]
    fixed_rpy, gt_rpy, fixed_rpy_error, fixed_geodesic = orientation_errors(
        fixed.rotation, gt_rotation
    )
    zero_rpy, _, zero_rpy_error, zero_geodesic = orientation_errors(
        zero.rotation, gt_rotation
    )
    reference_rpy, _, reference_rpy_error, reference_geodesic = orientation_errors(
        reference.rotation, gt_rotation
    )
    _, _, _, inekf_reference_difference = orientation_errors(
        fixed.rotation, reference.rotation
    )
    summaries = summarize_horizons(
        fixed.time,
        fixed_rpy_error,
        fixed_geodesic,
    )
    position_error = np.linalg.norm(fixed.position - gt_position, axis=1)
    index_10s = int(np.searchsorted(fixed.time, 10.0, side="right") - 1)
    position_error_10s = float(position_error[max(index_10s, 0)])
    position_error_final = float(position_error[-1])

    payload = {
        "configuration": {
            "sequence": str(data.source_dir),
            "calibration_seconds": args.calibration_seconds,
            "bias_source": args.bias_source,
            "evaluation_start": float(data.time[start]),
            "evaluation_end": float(data.time[end]),
            "uses_measurement_updates": False,
            "uses_gt_reinitialization": False,
            "imu_interval_sample": "current_endpoint",
        },
        "fixed_bias": {
            "gyro": fixed_bias.gyro.tolist(),
            "accel": fixed_bias.accel.tolist(),
        },
        "orientation_metrics": summaries,
        "zero_bias_full_metrics": metric_block(zero_rpy_error, zero_geodesic),
        "reference_full_metrics": metric_block(
            reference_rpy_error, reference_geodesic
        ),
        "inekf_reference_max_geodesic_difference_deg": float(
            np.max(inekf_reference_difference)
        ),
        "trajectory_diagnostic": {
            "position_error_at_10s_m": position_error_10s,
            "final_position_error_m": position_error_final,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    plot_rpy(
        fixed.time,
        gt_rpy,
        fixed_rpy,
        reference_rpy,
        figures / "01_roll_pitch_yaw.png",
    )
    plot_rpy_errors(
        fixed.time,
        fixed_rpy_error,
        zero_rpy_error,
        figures / "02_roll_pitch_yaw_error.png",
    )
    plot_geodesic_error(
        fixed.time,
        fixed_geodesic,
        zero_geodesic,
        reference_geodesic,
        figures / "03_so3_attitude_error.png",
    )
    plot_trajectory_diagnostic(
        fixed.time,
        gt_position,
        fixed,
        reference,
        figures / "04_trajectory_diagnostic.png",
    )
    write_report(
        output / "report.md",
        data,
        start,
        fixed_bias,
        args.bias_source,
        summaries,
        float(np.max(inekf_reference_difference)),
        position_error_10s,
        position_error_final,
    )
    print(f"Done. Results: {output}")


if __name__ == "__main__":
    main()
