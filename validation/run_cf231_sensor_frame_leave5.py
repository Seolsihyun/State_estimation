"""CF231 leave-run-5-out IMU dead reckoning in a common sensor frame.

The transferable bias and noise statistics come from the first static second
of training runs 3, 9, and 10 without using their mocap attitudes. Run 5 uses
the transferred bias, aligns roll/pitch from its initial accelerometer gravity
direction, receives initial GT yaw/position/velocity once, and then performs
prediction-only IMU dead reckoning.
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
from validation.cf231_loader import (
    GRAVITY_MAGNITUDE,
    SynchronizedRun,
    load_cf231_run,
    synchronize_to_imu,
)


@dataclass(frozen=True)
class SensorCalibration:
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gyro_sample_variance: np.ndarray
    accel_sample_variance: np.ndarray
    gyro_between_run_variance: np.ndarray
    accel_between_run_variance: np.ndarray
    sample_period_s: float
    per_run: dict[int, dict[str, object]]


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    imu_rotation: np.ndarray
    body_rotation: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    position_std: np.ndarray
    attitude_std_deg: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "cf231_leave_one_out" / "csv",
    )
    parser.add_argument("--training-runs", default="3,9,10")
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation"
        / "results"
        / "cf231_leave5_sensor_frame",
    )
    return parser.parse_args()


def robust_sample_variance(samples: np.ndarray, center: np.ndarray) -> np.ndarray:
    mad = np.median(np.abs(samples - center[None, :]), axis=0)
    return np.square(1.4826 * mad)


def estimate_sensor_calibration(
    dataset: Path,
    training_runs: list[int],
    static_seconds: float,
) -> SensorCalibration:
    gyro_biases: list[np.ndarray] = []
    accel_biases: list[np.ndarray] = []
    gyro_variances: list[np.ndarray] = []
    accel_variances: list[np.ndarray] = []
    sample_periods: list[float] = []
    per_run: dict[int, dict[str, object]] = {}
    gravity_sample = np.array([0.0, 0.0, GRAVITY_MAGNITUDE])

    for run_id in training_runs:
        run = synchronize_to_imu(load_cf231_run(dataset, run_id))
        static = run.time <= static_seconds
        sample_count = int(np.count_nonzero(static))
        if sample_count < 20:
            raise ValueError(f"Run {run_id} has only {sample_count} static samples.")

        accel_mean = np.mean(run.imu[static, 0:3], axis=0)
        gyro_mean = np.mean(run.imu[static, 3:6], axis=0)
        accel_bias = accel_mean - gravity_sample
        gyro_bias = gyro_mean
        accel_variance = robust_sample_variance(
            run.imu[static, 0:3],
            accel_mean,
        )
        gyro_variance = robust_sample_variance(
            run.imu[static, 3:6],
            gyro_mean,
        )
        speed = np.linalg.norm(run.velocity[static], axis=1)
        gyro_biases.append(gyro_bias)
        accel_biases.append(accel_bias)
        gyro_variances.append(gyro_variance)
        accel_variances.append(accel_variance)
        sample_periods.append(float(np.median(np.diff(run.time[static]))))
        per_run[run_id] = {
            "static_samples": sample_count,
            "static_duration_s": float(run.time[static][-1]),
            "gt_speed_median_m_s": float(np.median(speed)),
            "gt_speed_max_m_s": float(np.max(speed)),
            "raw_accel_mean_m_s2": accel_mean.tolist(),
            "gyro_bias_rad_s": gyro_bias.tolist(),
            "accel_bias_m_s2": accel_bias.tolist(),
            "gyro_sample_variance_rad2_s2": gyro_variance.tolist(),
            "accel_sample_variance_m2_s4": accel_variance.tolist(),
        }

    gyro_array = np.asarray(gyro_biases)
    accel_array = np.asarray(accel_biases)
    return SensorCalibration(
        gyro_bias=np.median(gyro_array, axis=0),
        accel_bias=np.median(accel_array, axis=0),
        gyro_sample_variance=np.median(np.asarray(gyro_variances), axis=0),
        accel_sample_variance=np.median(np.asarray(accel_variances), axis=0),
        gyro_between_run_variance=np.var(gyro_array, axis=0, ddof=1),
        accel_between_run_variance=np.var(accel_array, axis=0, ddof=1),
        sample_period_s=float(np.median(sample_periods)),
        per_run=per_run,
    )


def gravity_aligned_imu_rotation(
    run: SynchronizedRun,
    accel_bias: np.ndarray,
    static_seconds: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Initialize IMU-to-world roll/pitch from gravity and yaw from initial GT."""
    static = run.time <= static_seconds
    corrected_force = np.mean(run.imu[static, 0:3], axis=0) - accel_bias
    force_norm = float(np.linalg.norm(corrected_force))
    if force_norm < 0.5 * GRAVITY_MAGNITUDE:
        raise ValueError("Initial corrected accelerometer norm is not gravity-like.")
    roll = float(np.arctan2(corrected_force[1], corrected_force[2]))
    pitch = float(
        np.arctan2(
            -corrected_force[0],
            np.hypot(corrected_force[1], corrected_force[2]),
        )
    )
    gt_rpy = run.rotation[0].as_euler("xyz")
    yaw = float(gt_rpy[2])
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return rotation, {
        "corrected_initial_specific_force_m_s2": corrected_force.tolist(),
        "corrected_force_norm_m_s2": force_norm,
        "imu_initial_rpy_deg": np.rad2deg([roll, pitch, yaw]).tolist(),
        "gt_body_initial_rpy_deg": np.rad2deg(gt_rpy).tolist(),
    }


def make_filter(
    run: SynchronizedRun,
    calibration: SensorCalibration,
    initial_imu_rotation: np.ndarray,
    *,
    gravity_magnitude: float = GRAVITY_MAGNITUDE,
) -> HoonInvariantKalmanAnalytic15D:
    process_noise = np.zeros(15)
    process_noise[0:3] = np.maximum(
        calibration.gyro_sample_variance * calibration.sample_period_s,
        1.0e-12,
    )
    process_noise[3:6] = np.maximum(
        calibration.accel_sample_variance * calibration.sample_period_s,
        1.0e-12,
    )
    estimator = HoonInvariantKalmanAnalytic15D(
        mode="imu_only",
        motion_config={
            "gravity": [0.0, 0.0, -float(gravity_magnitude)],
            "jacobian_mode": "analytic",
            "process_noise_diag": process_noise,
            "update_biases": False,
            "covariance_ceiling": 1.0e18,
        },
        initialization_config={
            "mean": [0.0] * 6,
            "velocity_mean": [0.0] * 3,
            "cov_diag": [1.0e-8] * 9 + [0.0] * 6,
        },
    )
    estimator.Rot = initial_imu_rotation.copy()
    estimator.v = run.velocity[0].copy()
    estimator.p = run.position[0].copy()
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    estimator.set_fixed_imu_bias(
        calibration.gyro_bias,
        calibration.accel_bias,
    )
    estimator.P[9:12, 9:12] = np.diag(
        np.maximum(calibration.gyro_between_run_variance, 0.0)
    )
    estimator.P[12:15, 12:15] = np.diag(
        np.maximum(calibration.accel_between_run_variance, 0.0)
    )
    return estimator


def propagate(
    run: SynchronizedRun,
    calibration: SensorCalibration,
    initial_imu_rotation: np.ndarray,
    *,
    gravity_magnitude: float = GRAVITY_MAGNITUDE,
) -> Trajectory:
    estimator = make_filter(
        run,
        calibration,
        initial_imu_rotation,
        gravity_magnitude=gravity_magnitude,
    )
    initial_body_rotation = run.rotation[0].as_matrix()
    # C_BI maps IMU-frame vectors to the mocap body frame. It is used only to
    # express the estimated IMU orientation in the GT body frame for scoring.
    body_from_imu = initial_body_rotation.T @ initial_imu_rotation

    imu_rotations = [estimator.Rot.copy()]
    body_rotations = [estimator.Rot @ body_from_imu.T]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    position_std = [np.sqrt(np.clip(np.diag(estimator.P)[6:9], 0.0, None))]
    attitude_std = [
        np.rad2deg(np.sqrt(np.clip(np.diag(estimator.P)[0:3], 0.0, None)))
    ]
    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        if not 0.0 < dt < 0.5:
            raise ValueError(f"Unexpected IMU dt={dt} at sample {index}.")
        substeps = max(1, int(np.ceil(dt / 0.02)))
        for _ in range(substeps):
            estimator.predict(run.imu[index], dt / substeps)
        imu_rotations.append(estimator.Rot.copy())
        body_rotations.append(estimator.Rot @ body_from_imu.T)
        velocities.append(estimator.v.copy())
        positions.append(estimator.p.copy())
        position_std.append(
            np.sqrt(np.clip(np.diag(estimator.P)[6:9], 0.0, None))
        )
        attitude_std.append(
            np.rad2deg(np.sqrt(np.clip(np.diag(estimator.P)[0:3], 0.0, None)))
        )
    return Trajectory(
        time=run.time.copy(),
        imu_rotation=np.asarray(imu_rotations),
        body_rotation=np.asarray(body_rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
        position_std=np.asarray(position_std),
        attitude_std_deg=np.asarray(attitude_std),
    )


def calculate_metrics(
    trajectory: Trajectory,
    run: SynchronizedRun,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    position_error = np.linalg.norm(trajectory.position - run.position, axis=1)
    relative = np.einsum(
        "nji,njk->nik",
        run.rotation.as_matrix(),
        trajectory.body_rotation,
    )
    attitude_error = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    estimated_rpy = Rotation.from_matrix(trajectory.body_rotation).as_euler(
        "xyz",
        degrees=True,
    )
    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    rpy_error = (estimated_rpy - gt_rpy + 180.0) % 360.0 - 180.0
    result: dict[str, object] = {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_final_m": float(position_error[-1]),
        "position_max_m": float(np.max(position_error)),
        "so3_rmse_deg": float(np.sqrt(np.mean(np.square(attitude_error)))),
        "so3_p95_deg": float(np.percentile(attitude_error, 95)),
        "so3_final_deg": float(attitude_error[-1]),
        "rpy_rmse_deg": np.sqrt(np.mean(np.square(rpy_error), axis=0)).tolist(),
        "horizons": {},
    }
    for horizon in (1, 5, 10, 20, 30, 60, 120):
        if horizon > run.time[-1]:
            continue
        index = int(np.searchsorted(run.time, horizon, side="left"))
        result["horizons"][f"{horizon}s"] = {
            "position_error_m": float(position_error[index]),
            "so3_error_deg": float(attitude_error[index]),
        }
    return result, position_error, attitude_error


def plot_results(
    figures: Path,
    run: SynchronizedRun,
    trajectory: Trajectory,
    position_error: np.ndarray,
    attitude_error: np.ndarray,
    calibration: SensorCalibration,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    gt = run.position
    estimate = trajectory.position

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for axis, horizon in zip(axes.flat, (5, 10, 20, 30)):
        mask = run.time <= horizon
        axis.plot(gt[mask, 0], gt[mask, 1], "k", lw=1.8, label="GT")
        axis.plot(
            estimate[mask, 0],
            estimate[mask, 1],
            color="tab:blue",
            lw=1.5,
            label="sensor-frame IMU DR",
        )
        axis.scatter(gt[0, 0], gt[0, 1], c="green", s=35, zorder=3)
        axis.set_title(f"first {horizon} s")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Run 5 GT vs corrected sensor-frame IMU dead reckoning")
    fig.tight_layout()
    fig.savefig(figures / "01_horizon_trajectory_overlays.png", dpi=180)
    plt.close(fig)

    limit = 10.0
    crossing = np.flatnonzero(position_error >= limit)
    cutoff = int(crossing[0]) if crossing.size else len(run.time) - 1
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for axis in axes:
        axis.plot(gt[:, 0], gt[:, 1], "k", lw=1.7, label="GT full")
        axis.plot(
            estimate[: cutoff + 1, 0],
            estimate[: cutoff + 1, 1],
            color="tab:blue",
            lw=1.7,
            label=f"IMU DR 0–{run.time[cutoff]:.2f} s",
        )
        axis.scatter(gt[0, 0], gt[0, 1], c="green", s=45, zorder=3)
        axis.scatter(
            estimate[cutoff, 0],
            estimate[cutoff, 1],
            c="red",
            marker="x",
            s=70,
            zorder=3,
            label=f"{position_error[cutoff]:.2f} m error",
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Full GT figure eight + IMU DR before 10 m error")
    axes[0].legend()
    margin_x = max(0.15, 0.12 * np.ptp(gt[:, 0]))
    margin_y = max(0.15, 0.12 * np.ptp(gt[:, 1]))
    axes[1].set_xlim(np.min(gt[:, 0]) - margin_x, np.max(gt[:, 0]) + margin_x)
    axes[1].set_ylim(np.min(gt[:, 1]) - margin_y, np.max(gt[:, 1]) + margin_y)
    axes[1].set_title("Same overlay at GT scale")
    fig.tight_layout()
    fig.savefig(figures / "02_gt_full_vs_imu_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].semilogy(
        run.time,
        np.maximum(position_error, 1.0e-6),
        color="tab:blue",
    )
    axes[0].set_ylabel("position error [m]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(run.time, attitude_error, color="tab:orange")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "03_error_over_time.png", dpi=180)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    estimate_rpy = Rotation.from_matrix(trajectory.body_rotation).as_euler(
        "xyz",
        degrees=True,
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for index, label in enumerate(("roll", "pitch", "yaw")):
        axes[index].plot(run.time, gt_rpy[:, index], "k", lw=1.1, label="GT")
        axes[index].plot(
            run.time,
            estimate_rpy[:, index],
            color="tab:blue",
            lw=0.8,
            label="IMU DR",
        )
        axes[index].set_ylabel(f"{label} [deg]")
        axes[index].grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figures / "04_rpy.png", dpi=180)
    plt.close(fig)

    run_ids = sorted(calibration.per_run)
    accel_bias = np.asarray(
        [calibration.per_run[run_id]["accel_bias_m_s2"] for run_id in run_ids]
    )
    gyro_bias = np.asarray(
        [calibration.per_run[run_id]["gyro_bias_rad_s"] for run_id in run_ids]
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="row")
    for index, axis_name in enumerate(("x", "y", "z")):
        axes[0, index].plot(run_ids, gyro_bias[:, index], "o-")
        axes[0, index].axhline(calibration.gyro_bias[index], ls="--")
        axes[0, index].set_title(f"gyro {axis_name}")
        axes[0, index].grid(True, alpha=0.3)
        axes[1, index].plot(run_ids, accel_bias[:, index], "o-")
        axes[1, index].axhline(calibration.accel_bias[index], ls="--")
        axes[1, index].set_title(f"accelerometer {axis_name}")
        axes[1, index].set_xlabel("training run")
        axes[1, index].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("bias [rad/s]")
    axes[1, 0].set_ylabel("bias [m/s^2]")
    fig.tight_layout()
    fig.savefig(figures / "05_sensor_frame_bias.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    calibration = summary["calibration"]
    metrics = summary["metrics"]
    lines = [
        "# CF231 leave-run-5-out sensor-frame dead reckoning",
        "",
        "## Protocol",
        "",
        "- Training runs 3, 9, 10: first static second, raw IMU sensor frame.",
        "- No training-run mocap attitude is used to define accelerometer bias.",
        "- Run 5 roll/pitch: initial accelerometer gravity direction.",
        "- Run 5 yaw/position/velocity: one initial GT value.",
        "- Prediction only after initialization; no GT/PWM/AI/update.",
        "",
        "## Transfer calibration",
        "",
        f"- Gyro bias [rad/s]: {calibration['gyro_bias_rad_s']}",
        f"- Accelerometer bias [m/s^2]: {calibration['accel_bias_m_s2']}",
        f"- Gyro sample variance: {calibration['gyro_sample_variance_rad2_s2']}",
        f"- Accelerometer sample variance: {calibration['accel_sample_variance_m2_s4']}",
        "",
        "## Run 5 result",
        "",
        f"- Position RMSE: {metrics['position_rmse_m']:.6g} m",
        f"- Final position error: {metrics['position_final_m']:.6g} m",
        f"- SO(3) RMSE: {metrics['so3_rmse_deg']:.6g} deg",
        f"- SO(3) final: {metrics['so3_final_deg']:.6g} deg",
    ]
    for horizon, values in metrics["horizons"].items():
        lines.append(
            f"- {horizon}: position {values['position_error_m']:.6g} m, "
            f"SO(3) {values['so3_error_deg']:.6g} deg"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_runs = sorted(
        {int(value.strip()) for value in args.training_runs.split(",") if value.strip()}
    )
    if args.test_run in training_runs:
        raise ValueError("The held-out test run cannot be used for calibration.")

    print("Estimating sensor-frame calibration...")
    calibration = estimate_sensor_calibration(
        dataset,
        training_runs,
        args.static_seconds,
    )
    print("Loading held-out run and aligning initial gravity...")
    run = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    initial_rotation, alignment = gravity_aligned_imu_rotation(
        run,
        calibration.accel_bias,
        args.static_seconds,
    )
    print("Running continuous IMU-only propagation...")
    trajectory = propagate(run, calibration, initial_rotation)
    metrics, position_error, attitude_error = calculate_metrics(trajectory, run)

    summary = {
        "configuration": {
            "dataset": str(dataset),
            "training_runs": training_runs,
            "test_run": args.test_run,
            "static_seconds": args.static_seconds,
            "duration_s": float(run.time[-1]),
            "samples": int(run.time.size),
            "bias_frame": "raw IMU sensor frame",
            "initialization": (
                "roll/pitch from run-5 IMU gravity; yaw/p/v from one initial GT sample"
            ),
            "updates_after_initialization": [],
            "uses_training_pose_attitude_for_bias": False,
            "uses_test_gt_after_initialization": False,
            "uses_pwm": False,
            "uses_ai": False,
        },
        "calibration": {
            "gyro_bias_rad_s": calibration.gyro_bias.tolist(),
            "accel_bias_m_s2": calibration.accel_bias.tolist(),
            "gyro_sample_variance_rad2_s2": calibration.gyro_sample_variance.tolist(),
            "accel_sample_variance_m2_s4": calibration.accel_sample_variance.tolist(),
            "gyro_between_run_variance": calibration.gyro_between_run_variance.tolist(),
            "accel_between_run_variance": calibration.accel_between_run_variance.tolist(),
            "sample_period_s": calibration.sample_period_s,
            "per_run": calibration.per_run,
        },
        "run5_initial_alignment": alignment,
        "metrics": metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=run.time,
        gt_position=run.position,
        gt_body_rotation=run.rotation.as_matrix(),
        estimated_position=trajectory.position,
        estimated_velocity=trajectory.velocity,
        estimated_imu_rotation=trajectory.imu_rotation,
        estimated_body_rotation=trajectory.body_rotation,
        position_std=trajectory.position_std,
        attitude_std_deg=trajectory.attitude_std_deg,
    )
    plot_results(
        output / "figures",
        run,
        trajectory,
        position_error,
        attitude_error,
        calibration,
    )
    write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
