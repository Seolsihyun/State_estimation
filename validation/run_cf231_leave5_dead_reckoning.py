"""Leave-run-5-out CF231 IMU-only dead-reckoning validation.

Runs 1-4 and 6-11 are calibration candidates. Runs without IMU data or with
too few low-dynamics samples are recorded and skipped. Their mocap ground
truth is used only to estimate one transferable fixed IMU bias and residual
variance. Run 5 receives one GT R/v/p initialization and then uses IMU only.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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
    available_run_ids,
    imu_gt_residuals,
    load_cf231_run,
    low_dynamics_mask,
    synchronize_to_imu,
)


@dataclass(frozen=True)
class Calibration:
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gyro_noise_variance: np.ndarray
    accel_noise_variance: np.ndarray
    gyro_between_run_variance: np.ndarray
    accel_between_run_variance: np.ndarray
    median_sample_period_s: float
    per_run: dict[int, dict[str, object]]
    skipped_runs: dict[int, str]


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    rotation: np.ndarray
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
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument(
        "--training-runs",
        type=str,
        default="3,9,10",
        help=(
            "Comma-separated calibration runs selected after GT quality "
            "screening. Default: 3,9,10."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation"
        / "results"
        / "cf231_leave5_valid_runs_3_9_10",
    )
    parser.add_argument("--min-calibration-samples", type=int, default=100)
    return parser.parse_args()


def robust_variance(samples: np.ndarray, center: np.ndarray) -> np.ndarray:
    absolute_deviation = np.abs(samples - center[None, :])
    mad = np.median(absolute_deviation, axis=0)
    return np.square(1.4826 * mad)


def estimate_leave_one_out_calibration(
    dataset: Path,
    test_run: int,
    min_samples: int,
    training_runs: list[int] | None = None,
) -> Calibration:
    per_run: dict[int, dict[str, object]] = {}
    skipped: dict[int, str] = {}
    gyro_biases: list[np.ndarray] = []
    accel_biases: list[np.ndarray] = []
    gyro_variances: list[np.ndarray] = []
    accel_variances: list[np.ndarray] = []
    sample_periods: list[float] = []

    for run_id in available_run_ids(dataset):
        if run_id == test_run:
            skipped[run_id] = "held out for validation"
            continue
        if training_runs is not None and run_id not in training_runs:
            skipped[run_id] = "not selected by manual GT-quality screening"
            continue
        try:
            synchronized = synchronize_to_imu(load_cf231_run(dataset, run_id))
        except (FileNotFoundError, ValueError) as error:
            skipped[run_id] = str(error)
            continue
        mask = low_dynamics_mask(synchronized)
        sample_count = int(np.count_nonzero(mask))
        if sample_count < min_samples:
            skipped[run_id] = (
                f"only {sample_count} low-dynamics samples; "
                f"minimum is {min_samples}"
            )
            continue

        gyro_residual, accel_residual = imu_gt_residuals(synchronized, mask)
        gyro_bias = np.median(gyro_residual, axis=0)
        accel_bias = np.median(accel_residual, axis=0)
        gyro_variance = robust_variance(gyro_residual, gyro_bias)
        accel_variance = robust_variance(accel_residual, accel_bias)
        gyro_biases.append(gyro_bias)
        accel_biases.append(accel_bias)
        gyro_variances.append(gyro_variance)
        accel_variances.append(accel_variance)
        sample_periods.append(float(np.median(np.diff(synchronized.time))))
        per_run[run_id] = {
            "synchronized_samples": int(synchronized.time.size),
            "duration_s": float(synchronized.time[-1]),
            "low_dynamics_samples": sample_count,
            "median_sample_period_s": sample_periods[-1],
            "gyro_bias_rad_s": gyro_bias.tolist(),
            "accel_bias_m_s2": accel_bias.tolist(),
            "gyro_noise_variance_rad2_s2": gyro_variance.tolist(),
            "accel_noise_variance_m2_s4": accel_variance.tolist(),
        }

    if len(gyro_biases) < 2:
        raise RuntimeError("At least two usable training runs are required.")
    gyro_bias_array = np.asarray(gyro_biases)
    accel_bias_array = np.asarray(accel_biases)
    return Calibration(
        # Run-balanced medians prevent a long run from dominating the estimate.
        gyro_bias=np.median(gyro_bias_array, axis=0),
        accel_bias=np.median(accel_bias_array, axis=0),
        gyro_noise_variance=np.median(np.asarray(gyro_variances), axis=0),
        accel_noise_variance=np.median(np.asarray(accel_variances), axis=0),
        gyro_between_run_variance=np.var(gyro_bias_array, axis=0, ddof=1),
        accel_between_run_variance=np.var(accel_bias_array, axis=0, ddof=1),
        median_sample_period_s=float(np.median(sample_periods)),
        per_run=per_run,
        skipped_runs=skipped,
    )


def test_run_bias_diagnostic(run: SynchronizedRun) -> tuple[np.ndarray, np.ndarray, int]:
    mask = low_dynamics_mask(run)
    gyro_residual, accel_residual = imu_gt_residuals(run, mask)
    if gyro_residual.shape[0] < 10:
        raise RuntimeError("Held-out run has too few diagnostic low-dynamics samples.")
    return (
        np.median(gyro_residual, axis=0),
        np.median(accel_residual, axis=0),
        int(gyro_residual.shape[0]),
    )


def make_filter(
    run: SynchronizedRun,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    calibration: Calibration,
) -> HoonInvariantKalmanAnalytic15D:
    process_noise = np.zeros(15)
    # The filter discretizes Qc as Qc * dt. Convert per-sample residual
    # variance to an approximate continuous white-noise PSD by multiplying by
    # the representative sample period.
    process_noise[0:3] = np.maximum(
        calibration.gyro_noise_variance * calibration.median_sample_period_s,
        1.0e-12,
    )
    process_noise[3:6] = np.maximum(
        calibration.accel_noise_variance * calibration.median_sample_period_s,
        1.0e-12,
    )
    estimator = HoonInvariantKalmanAnalytic15D(
        mode="imu_only",
        motion_config={
            "gravity": [0.0, 0.0, -GRAVITY_MAGNITUDE],
            "jacobian_mode": "analytic",
            "process_noise_diag": process_noise,
            "update_biases": False,
            # Long prediction-only runs legitimately produce very large
            # covariance. Avoid the base class's 1e6 diagnostic ceiling.
            "covariance_ceiling": 1.0e18,
        },
        initialization_config={
            "mean": [0.0] * 6,
            "velocity_mean": [0.0] * 3,
            "cov_diag": [1.0e-8] * 9 + [0.0] * 6,
        },
    )
    estimator.Rot = run.rotation[0].as_matrix().copy()
    estimator.v = run.velocity[0].copy()
    estimator.p = run.position[0].copy()
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    estimator.set_fixed_imu_bias(gyro_bias, accel_bias)
    # The transferred bias is constant, but its value is uncertain on a new
    # run. Preserve between-run calibration variance in P without allowing the
    # mean bias to evolve.
    estimator.P[9:12, 9:12] = np.diag(
        np.maximum(calibration.gyro_between_run_variance, 0.0)
    )
    estimator.P[12:15, 12:15] = np.diag(
        np.maximum(calibration.accel_between_run_variance, 0.0)
    )
    return estimator


def propagate(
    run: SynchronizedRun,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    calibration: Calibration,
) -> Trajectory:
    estimator = make_filter(run, gyro_bias, accel_bias, calibration)
    rotations = [estimator.Rot.copy()]
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
        # Preserve elapsed time across short logger gaps without inventing a
        # GT-derived input. The current IMU sample is held constant in substeps.
        substeps = max(1, int(np.ceil(dt / 0.02)))
        substep_dt = dt / substeps
        for _ in range(substeps):
            estimator.predict(run.imu[index], substep_dt)
        rotations.append(estimator.Rot.copy())
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
        rotation=np.asarray(rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
        position_std=np.asarray(position_std),
        attitude_std_deg=np.asarray(attitude_std),
    )


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def metrics(trajectory: Trajectory, run: SynchronizedRun) -> dict[str, object]:
    position_error_vector = trajectory.position - run.position
    position_error = np.linalg.norm(position_error_vector, axis=1)
    relative_rotation = np.einsum(
        "nji,njk->nik",
        run.rotation.as_matrix(),
        trajectory.rotation,
    )
    so3_error = np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude())
    estimated_rpy = Rotation.from_matrix(trajectory.rotation).as_euler(
        "xyz", degrees=True
    )
    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    rpy_error = wrap_degrees(estimated_rpy - gt_rpy)
    position_three_sigma = 3.0 * np.linalg.norm(trajectory.position_std, axis=1)
    attitude_three_sigma = 3.0 * np.linalg.norm(
        trajectory.attitude_std_deg,
        axis=1,
    )
    result: dict[str, object] = {
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_error)))),
        "position_median_m": float(np.median(position_error)),
        "position_max_m": float(np.max(position_error)),
        "position_final_m": float(position_error[-1]),
        "so3_rmse_deg": float(np.sqrt(np.mean(np.square(so3_error)))),
        "so3_p95_deg": float(np.percentile(so3_error, 95)),
        "so3_max_deg": float(np.max(so3_error)),
        "so3_final_deg": float(so3_error[-1]),
        "rpy_rmse_deg": np.sqrt(np.mean(np.square(rpy_error), axis=0)).tolist(),
        "position_3sigma_coverage_fraction": float(
            np.mean(position_error <= position_three_sigma)
        ),
        "attitude_3sigma_coverage_fraction": float(
            np.mean(so3_error <= attitude_three_sigma)
        ),
        "position_3sigma_final_m": float(position_three_sigma[-1]),
        "attitude_3sigma_final_deg": float(attitude_three_sigma[-1]),
        "horizons": {},
    }
    for horizon in (1, 5, 10, 30, 60, 120):
        if horizon > trajectory.time[-1]:
            continue
        index = int(np.searchsorted(trajectory.time, horizon, side="left"))
        result["horizons"][f"{horizon}s"] = {
            "position_error_m": float(position_error[index]),
            "so3_error_deg": float(so3_error[index]),
        }
    return result


def plot_results(
    output: Path,
    run: SynchronizedRun,
    trajectories: dict[str, Trajectory],
    case_names: dict[str, str],
    calibration: Calibration,
    test_gyro_bias: np.ndarray,
    test_accel_bias: np.ndarray,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    colors = {
        "transferred": "tab:blue",
        "zero": "0.55",
        "test5_diagnostic": "tab:orange",
    }

    fig, axis = plt.subplots(figsize=(9, 8))
    axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=1.5, label="GT")
    for key, trajectory in trajectories.items():
        axis.plot(
            trajectory.position[:, 0],
            trajectory.position[:, 1],
            color=colors[key],
            lw=1.0,
            label=case_names[key],
        )
    axis.scatter(run.position[0, 0], run.position[0, 1], c="green", s=45, label="start")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Run 5 continuous IMU-only dead reckoning")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "01_xy_trajectory.png", dpi=180)
    plt.close(fig)

    for horizon in (10, 30):
        horizon_mask = run.time <= horizon
        fig, axis = plt.subplots(figsize=(9, 8))
        axis.plot(
            run.position[horizon_mask, 0],
            run.position[horizon_mask, 1],
            "k",
            lw=1.5,
            label="GT",
        )
        for key, trajectory in trajectories.items():
            axis.plot(
                trajectory.position[horizon_mask, 0],
                trajectory.position[horizon_mask, 1],
                color=colors[key],
                lw=1.0,
                label=case_names[key],
            )
        axis.scatter(
            run.position[0, 0],
            run.position[0, 1],
            c="green",
            s=45,
            label="start",
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(f"Run 5 IMU-only trajectory: first {horizon} s")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"01_{horizon:03d}s_xy_trajectory.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for key, trajectory in trajectories.items():
        error = np.linalg.norm(trajectory.position - run.position, axis=1)
        axis.semilogy(
            trajectory.time,
            np.maximum(error, 1.0e-5),
            color=colors[key],
            lw=1.0,
            label=case_names[key],
        )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("position error [m], log scale")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "02_position_error.png", dpi=180)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    estimated_rpy = Rotation.from_matrix(
        trajectories["transferred"].rotation
    ).as_euler("xyz", degrees=True)
    labels = ("roll", "pitch", "yaw")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis_index, axis in enumerate(axes):
        axis.plot(run.time, gt_rpy[:, axis_index], "k", lw=1.2, label="GT")
        axis.plot(
            run.time,
            estimated_rpy[:, axis_index],
            color=colors["transferred"],
            lw=0.9,
            label=case_names["transferred"],
        )
        axis.set_ylabel(f"{labels[axis_index]} [deg]")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output / "03_rpy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    gt_rotation = run.rotation.as_matrix()
    for key, trajectory in trajectories.items():
        relative = np.einsum("nji,njk->nik", gt_rotation, trajectory.rotation)
        error = np.rad2deg(Rotation.from_matrix(relative).magnitude())
        axis.plot(
            run.time,
            error,
            color=colors[key],
            lw=1.0,
            label=case_names[key],
        )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("SO(3) attitude error [deg]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "04_so3_error.png", dpi=180)
    plt.close(fig)

    used_runs = sorted(calibration.per_run)
    gyro_bias = np.asarray(
        [calibration.per_run[run_id]["gyro_bias_rad_s"] for run_id in used_runs]
    )
    accel_bias = np.asarray(
        [calibration.per_run[run_id]["accel_bias_m_s2"] for run_id in used_runs]
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="row")
    axis_names = ("x", "y", "z")
    for axis_index in range(3):
        axes[0, axis_index].plot(used_runs, gyro_bias[:, axis_index], "o-", label="train runs")
        axes[0, axis_index].axhline(
            calibration.gyro_bias[axis_index],
            color="tab:blue",
            ls="--",
            label="transferred",
        )
        axes[0, axis_index].axhline(
            test_gyro_bias[axis_index],
            color="tab:orange",
            ls=":",
            label="run 5 diagnostic",
        )
        axes[0, axis_index].set_title(f"gyro {axis_names[axis_index]}")
        axes[0, axis_index].grid(True, alpha=0.3)
        axes[1, axis_index].plot(used_runs, accel_bias[:, axis_index], "o-")
        axes[1, axis_index].axhline(
            calibration.accel_bias[axis_index],
            color="tab:blue",
            ls="--",
        )
        axes[1, axis_index].axhline(
            test_accel_bias[axis_index],
            color="tab:orange",
            ls=":",
        )
        axes[1, axis_index].set_title(f"accelerometer {axis_names[axis_index]}")
        axes[1, axis_index].set_xlabel("run")
        axes[1, axis_index].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("bias [rad/s]")
    axes[1, 0].set_ylabel("bias [m/s^2]")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "05_bias_transfer.png", dpi=180)
    plt.close(fig)

    transferred = trajectories["transferred"]
    position_error = np.linalg.norm(transferred.position - run.position, axis=1)
    relative = np.einsum(
        "nji,njk->nik",
        run.rotation.as_matrix(),
        transferred.rotation,
    )
    attitude_error = np.rad2deg(Rotation.from_matrix(relative).magnitude())
    position_three_sigma = 3.0 * np.linalg.norm(
        transferred.position_std,
        axis=1,
    )
    attitude_three_sigma = 3.0 * np.linalg.norm(
        transferred.attitude_std_deg,
        axis=1,
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].semilogy(
        run.time,
        np.maximum(position_error, 1.0e-6),
        color="tab:red",
        label="position error",
    )
    axes[0].semilogy(
        run.time,
        np.maximum(position_three_sigma, 1.0e-6),
        color="tab:blue",
        ls="--",
        label="propagated 3 sigma",
    )
    axes[0].set_ylabel("position [m], log scale")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        run.time,
        attitude_error,
        color="tab:red",
        label="SO(3) error",
    )
    axes[1].plot(
        run.time,
        attitude_three_sigma,
        color="tab:blue",
        ls="--",
        label="propagated 3 sigma",
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("attitude [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "06_uncertainty_consistency.png", dpi=180)
    plt.close(fig)

    # Each full trajectory gets its own metric axis. Putting all four on one
    # axis hides the metre-scale GT underneath kilometre-scale inertial drift.
    full_paths = {
        "GT": run.position,
        case_names["transferred"]: trajectories["transferred"].position,
        case_names["zero"]: trajectories["zero"].position,
        case_names["test5_diagnostic"]: trajectories["test5_diagnostic"].position,
    }
    norm = Normalize(vmin=float(run.time[0]), vmax=float(run.time[-1]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for axis, (title, position) in zip(axes.flat, full_paths.items()):
        xy = position[:, :2]
        segments = np.stack((xy[:-1], xy[1:]), axis=1)
        collection = LineCollection(
            segments,
            cmap="viridis",
            norm=norm,
            linewidth=1.2,
        )
        collection.set_array(run.time[:-1])
        axis.add_collection(collection)
        axis.scatter(xy[0, 0], xy[0, 1], c="limegreen", s=45, zorder=3, label="start")
        axis.scatter(xy[-1, 0], xy[-1, 1], c="red", s=45, zorder=3, label="end")
        axis.autoscale()
        axis.margins(0.08)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title}\n"
            f"range: dx={np.ptp(xy[:, 0]):.3g} m, dy={np.ptp(xy[:, 1]):.3g} m"
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.03,
    )
    colorbar.set_label("time [s]")
    fig.suptitle(
        "Run 5 full 181.3 s trajectory — independent metric scales",
        fontsize=15,
    )
    fig.savefig(
        output / "07_full_trajectory_independent_scales.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    gt_xy = run.position[:, :2]
    fig, axis = plt.subplots(figsize=(10, 8))
    segments = np.stack((gt_xy[:-1], gt_xy[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=1.5,
    )
    collection.set_array(run.time[:-1])
    axis.add_collection(collection)
    axis.scatter(gt_xy[0, 0], gt_xy[0, 1], c="limegreen", s=55, zorder=3, label="start")
    axis.scatter(gt_xy[-1, 0], gt_xy[-1, 1], c="red", s=55, zorder=3, label="end")
    axis.autoscale()
    axis.margins(0.08)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Run 5 full GT figure-eight trajectory")
    axis.grid(True, alpha=0.3)
    axis.legend()
    colorbar = fig.colorbar(collection, ax=axis)
    colorbar.set_label("time [s]")
    fig.tight_layout()
    fig.savefig(output / "08_full_gt_figure8.png", dpi=180)
    plt.close(fig)

    # Presentation overlay: retain the complete GT figure eight, but stop the
    # IMU curve at a declared error threshold so kilometre-scale divergence
    # does not visually erase the metre-scale reference.
    transferred_xy = trajectories["transferred"].position[:, :2]
    transferred_error = np.linalg.norm(
        trajectories["transferred"].position - run.position,
        axis=1,
    )
    error_limit_m = 10.0
    crossings = np.flatnonzero(transferred_error >= error_limit_m)
    cutoff = int(crossings[0]) if crossings.size else len(run.time) - 1
    displayed_xy = transferred_xy[: cutoff + 1]
    cutoff_time = float(run.time[cutoff])
    cutoff_error = float(transferred_error[cutoff])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    for axis in axes:
        axis.plot(
            gt_xy[:, 0],
            gt_xy[:, 1],
            color="black",
            lw=1.7,
            label="GT: full 181.3 s",
        )
        axis.plot(
            displayed_xy[:, 0],
            displayed_xy[:, 1],
            color="tab:blue",
            lw=1.7,
            label=f"IMU DR: 0–{cutoff_time:.2f} s",
        )
        axis.scatter(
            gt_xy[0, 0],
            gt_xy[0, 1],
            c="limegreen",
            edgecolor="black",
            s=55,
            zorder=4,
            label="start",
        )
        axis.scatter(
            displayed_xy[-1, 0],
            displayed_xy[-1, 1],
            c="red",
            marker="x",
            s=75,
            zorder=4,
            label=f"10 m cutoff ({cutoff_error:.2f} m)",
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Full GT figure eight + early IMU dead reckoning")
    axes[0].legend(loc="best")

    gt_margin_x = max(0.15, 0.12 * np.ptp(gt_xy[:, 0]))
    gt_margin_y = max(0.15, 0.12 * np.ptp(gt_xy[:, 1]))
    axes[1].set_xlim(
        float(np.min(gt_xy[:, 0]) - gt_margin_x),
        float(np.max(gt_xy[:, 0]) + gt_margin_x),
    )
    axes[1].set_ylim(
        float(np.min(gt_xy[:, 1]) - gt_margin_y),
        float(np.max(gt_xy[:, 1]) + gt_margin_y),
    )
    axes[1].set_title("Same overlay zoomed to the GT scale")
    fig.suptitle(
        "Run 5: IMU trajectory is displayed only until its first 10 m error",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output / "09_gt_vs_imu_dead_reckoning_overlay.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    calibration = summary["calibration"]
    transferred = summary["test_metrics"]["transferred"]
    zero = summary["test_metrics"]["zero"]
    diagnostic = summary["test_metrics"]["test5_diagnostic"]
    bias_diagnostic = summary["held_out_bias_diagnostic_not_used_by_main"]
    lines = [
        "# CF231 leave-run-5-out fixed-bias dead reckoning",
        "",
        "## Protocol",
        "",
        "- Bias and residual variance use only eligible runs other than run 5.",
        "- Run 5 uses one GT R/v/p initialization, followed by IMU prediction only.",
        "- Run 5 position and orientation GT are used only for evaluation.",
        "- No position, velocity, attitude, PWM, ZUPT, GPS, or AI update is used.",
        "",
        "## Transfer calibration",
        "",
        f"- Used training runs: {calibration['used_runs']}",
        f"- Skipped runs: {calibration['skipped_runs']}",
        f"- Gyro bias [rad/s]: {calibration['gyro_bias_rad_s']}",
        f"- Accelerometer bias [m/s^2]: {calibration['accel_bias_m_s2']}",
        f"- Gyro noise variance: {calibration['gyro_noise_variance_rad2_s2']}",
        f"- Accelerometer noise variance: {calibration['accel_noise_variance_m2_s4']}",
        f"- Median training sample period: {calibration['median_sample_period_s']:.6g} s",
        f"- Gyro process-noise PSD: {calibration['gyro_process_noise_psd']}",
        f"- Accelerometer process-noise PSD: {calibration['accel_process_noise_psd']}",
        f"- Transferred minus run-5 accelerometer bias: "
        f"{bias_diagnostic['transferred_minus_test_accel_bias_m_s2']} m/s^2",
        "",
        "## Held-out run 5 result",
        "",
        f"- Duration: {summary['configuration']['test_duration_s']:.6g} s",
        f"- Position RMSE: {transferred['position_rmse_m']:.6g} m",
        f"- Final position error: {transferred['position_final_m']:.6g} m",
        f"- SO(3) RMSE: {transferred['so3_rmse_deg']:.6g} deg",
        f"- SO(3) final: {transferred['so3_final_deg']:.6g} deg",
        f"- Position 3-sigma coverage: "
        f"{100.0 * transferred['position_3sigma_coverage_fraction']:.3f}%",
        f"- Attitude 3-sigma coverage: "
        f"{100.0 * transferred['attitude_3sigma_coverage_fraction']:.3f}%",
        f"- Position error at 10 s: {transferred['horizons']['10s']['position_error_m']:.6g} m",
        f"- Position error at 30 s: {transferred['horizons']['30s']['position_error_m']:.6g} m",
        f"- Zero-bias final position error: {zero['position_final_m']:.6g} m",
        f"- Run-5-bias diagnostic final position error: {diagnostic['position_final_m']:.6g} m",
        f"- Run-5-bias diagnostic SO(3) RMSE: {diagnostic['so3_rmse_deg']:.6g} deg",
        "",
        "The run-5-bias case is diagnostic only and is not the leave-one-out result.",
        "The learned variance changes covariance/uncertainty, but cannot change the",
        "prediction-only nominal trajectory without a measurement update.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    training_runs = sorted(
        {
            int(value.strip())
            for value in args.training_runs.split(",")
            if value.strip()
        }
    )
    if args.test_run in training_runs:
        raise ValueError("The held-out test run cannot be a training run.")
    if not training_runs:
        raise ValueError("At least one training run must be supplied.")

    print("Estimating leave-one-run-out bias and variance...")
    calibration = estimate_leave_one_out_calibration(
        dataset,
        args.test_run,
        args.min_calibration_samples,
        training_runs,
    )
    print("Loading held-out run...")
    held_out = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    held_out_dt = np.diff(held_out.time)
    test_gyro_bias, test_accel_bias, test_bias_samples = test_run_bias_diagnostic(
        held_out
    )

    cases = {
        "transferred": (calibration.gyro_bias, calibration.accel_bias),
        "zero": (np.zeros(3), np.zeros(3)),
        "test5_diagnostic": (test_gyro_bias, test_accel_bias),
    }
    case_names = {
        "transferred": "runs != 5 fixed bias",
        "zero": "zero bias",
        "test5_diagnostic": "run-5 bias (diagnostic only)",
    }
    trajectories: dict[str, Trajectory] = {}
    test_metrics: dict[str, object] = {}
    for key, (gyro_bias, accel_bias) in cases.items():
        print(f"Propagating {key}...")
        trajectory = propagate(
            held_out,
            gyro_bias,
            accel_bias,
            calibration,
        )
        trajectories[key] = trajectory
        test_metrics[key] = metrics(trajectory, held_out)

    summary = {
        "configuration": {
            "dataset": str(dataset),
            "test_run": args.test_run,
            "requested_training_runs": training_runs,
            "training_run_selection": (
                "manual GT-quality screen: complete/clean figure-eight XY and "
                "stable altitude, with synchronized IMU available"
            ),
            "test_duration_s": float(held_out.time[-1]),
            "test_samples": int(held_out.time.size),
            "median_imu_dt_s": float(np.median(held_out_dt)),
            "max_imu_dt_s": float(np.max(held_out_dt)),
            "imu_gaps_over_20ms": int(np.count_nonzero(held_out_dt > 0.02)),
            "imu_units": {
                "accelerometer": "m/s^2 (converted from Crazyflie g)",
                "gyroscope": "rad/s (converted from Crazyflie deg/s)",
            },
            "initialization": "one held-out GT R, v, p at first synchronized sample",
            "updates_after_initialization": [],
            "uses_test_run_for_bias_or_variance": False,
            "uses_pwm": False,
            "uses_ai": False,
        },
        "calibration": {
            "used_runs": sorted(calibration.per_run),
            "skipped_runs": calibration.skipped_runs,
            "estimator": "run-balanced median of low-dynamics GT/IMU residuals",
            "gyro_bias_rad_s": calibration.gyro_bias.tolist(),
            "accel_bias_m_s2": calibration.accel_bias.tolist(),
            "gyro_noise_variance_rad2_s2": calibration.gyro_noise_variance.tolist(),
            "accel_noise_variance_m2_s4": calibration.accel_noise_variance.tolist(),
            "median_sample_period_s": calibration.median_sample_period_s,
            "gyro_process_noise_psd": (
                calibration.gyro_noise_variance
                * calibration.median_sample_period_s
            ).tolist(),
            "accel_process_noise_psd": (
                calibration.accel_noise_variance
                * calibration.median_sample_period_s
            ).tolist(),
            "gyro_between_run_bias_variance": calibration.gyro_between_run_variance.tolist(),
            "accel_between_run_bias_variance": calibration.accel_between_run_variance.tolist(),
            "per_run": calibration.per_run,
        },
        "held_out_bias_diagnostic_not_used_by_main": {
            "sample_count": test_bias_samples,
            "gyro_bias_rad_s": test_gyro_bias.tolist(),
            "accel_bias_m_s2": test_accel_bias.tolist(),
            "transferred_minus_test_gyro_bias_rad_s": (
                calibration.gyro_bias - test_gyro_bias
            ).tolist(),
            "transferred_minus_test_accel_bias_m_s2": (
                calibration.accel_bias - test_accel_bias
            ).tolist(),
        },
        "test_metrics": test_metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=held_out.time,
        gt_position=held_out.position,
        gt_rotation=held_out.rotation.as_matrix(),
        transferred_position=trajectories["transferred"].position,
        transferred_rotation=trajectories["transferred"].rotation,
        transferred_position_std=trajectories["transferred"].position_std,
        transferred_attitude_std_deg=trajectories["transferred"].attitude_std_deg,
        zero_position=trajectories["zero"].position,
        zero_rotation=trajectories["zero"].rotation,
        test5_diagnostic_position=trajectories["test5_diagnostic"].position,
        test5_diagnostic_rotation=trajectories["test5_diagnostic"].rotation,
    )
    plot_results(
        figures,
        held_out,
        trajectories,
        case_names,
        calibration,
        test_gyro_bias,
        test_accel_bias,
    )
    write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
