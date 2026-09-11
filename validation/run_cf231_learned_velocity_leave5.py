"""Train on CF231 runs other than 5 and evaluate run 5 with IMU input only.

The learned model maps a two-second raw-IMU window to velocity expressed in a
yaw-heading frame.  During held-out run-5 inference, the velocity is rotated
to the world frame with the InEKF attitude estimate and supplied through the
existing InEKF velocity measurement update.  Run-5 ground truth is loaded only
after propagation to score and plot the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Hoon_invariant_inekf as lie
from validation.cf231_loader import (
    SynchronizedRun,
    load_cf231_run,
    synchronize_to_imu,
)
from validation.run_cf231_sensor_frame_leave5 import (
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
    gravity_aligned_imu_rotation,
    make_filter,
    propagate,
)


@dataclass(frozen=True)
class WindowDataset:
    features: np.ndarray
    target_heading_velocity: np.ndarray
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "cf231_leave_one_out" / "csv",
    )
    parser.add_argument("--training-runs", default="3,4,9,10")
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--bias-runs", default="3,9,10")
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--update-stride", type=int, default=10)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--velocity-noise-scale",
        type=float,
        default=1.0,
        help="Scale applied to leave-one-run-out velocity residual variance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation"
        / "results"
        / "cf231_leave5_learned_velocity",
    )
    return parser.parse_args()


def parse_run_ids(text: str) -> list[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def heading_velocity(run: SynchronizedRun) -> np.ndarray:
    yaw = run.rotation.as_euler("xyz")[:, 2]
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    velocity = run.velocity
    return np.column_stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
            velocity[:, 2],
        )
    )


def training_quality_mask(run: SynchronizedRun) -> np.ndarray:
    """Reject mocap derivative failures without using a held-out run."""
    speed = np.linalg.norm(run.velocity, axis=1)
    acceleration = np.linalg.norm(run.acceleration, axis=1)
    angular_rate = np.linalg.norm(run.angular_velocity_body, axis=1)
    position_step = np.r_[
        0.0,
        np.linalg.norm(np.diff(run.position, axis=0), axis=1),
    ]
    return (
        np.all(np.isfinite(run.imu), axis=1)
        & np.all(np.isfinite(run.velocity), axis=1)
        & (speed < 4.0)
        & (acceleration < 15.0)
        & (angular_rate < 4.0)
        & (position_step < 0.05)
    )


def feature_vector(window: np.ndarray, bins: int) -> np.ndarray:
    if window.shape[0] % bins:
        raise ValueError("window-samples must be divisible by bins.")
    grouped = window.reshape(bins, window.shape[0] // bins, 6)
    return np.r_[
        grouped.mean(axis=1).ravel(),
        grouped.std(axis=1).ravel(),
        window[-1],
        window.mean(axis=0),
        window.std(axis=0),
        window.min(axis=0),
        window.max(axis=0),
    ]


def inference_windows(
    run: SynchronizedRun,
    window_samples: int,
    bins: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(window_samples - 1, run.time.size, stride, dtype=int)
    features = np.asarray(
        [
            feature_vector(
                run.imu[index - window_samples + 1 : index + 1],
                bins,
            )
            for index in indices
        ]
    )
    return features, indices


def training_windows(
    run: SynchronizedRun,
    window_samples: int,
    bins: int,
    stride: int,
) -> WindowDataset:
    features, indices = inference_windows(run, window_samples, bins, stride)
    quality = training_quality_mask(run)
    keep = np.asarray(
        [
            quality[index]
            and np.mean(quality[index - window_samples + 1 : index + 1]) >= 0.9
            for index in indices
        ]
    )
    target = heading_velocity(run)[indices]
    keep &= np.all(np.isfinite(target), axis=1)
    return WindowDataset(
        features=features[keep],
        target_heading_velocity=target[keep],
        indices=indices[keep],
    )


def model_parameters(trees: int, seed: int = 42) -> dict[str, object]:
    return {
        "n_estimators": trees,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "n_jobs": -1,
        "random_state": seed,
    }


def cross_validate(
    datasets: dict[int, WindowDataset],
    trees: int,
) -> tuple[dict[int, dict[str, object]], np.ndarray, np.ndarray]:
    per_run: dict[int, dict[str, object]] = {}
    residuals: list[np.ndarray] = []
    for held_out in sorted(datasets):
        train_features = np.vstack(
            [
                dataset.features
                for run_id, dataset in datasets.items()
                if run_id != held_out
            ]
        )
        train_target = np.vstack(
            [
                dataset.target_heading_velocity
                for run_id, dataset in datasets.items()
                if run_id != held_out
            ]
        )
        model = ExtraTreesRegressor(
            **model_parameters(max(80, trees // 2), seed=42 + held_out)
        )
        model.fit(train_features, train_target)
        prediction = model.predict(datasets[held_out].features)
        residual = prediction - datasets[held_out].target_heading_velocity
        residuals.append(residual)
        per_run[held_out] = {
            "samples": int(residual.shape[0]),
            "rmse_m_s": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
            "vector_rmse_m_s": float(np.sqrt(np.mean(np.square(residual)))),
        }
    pooled = np.vstack(residuals)
    return (
        per_run,
        np.mean(np.square(pooled), axis=0),
        np.mean(pooled, axis=0),
    )


def train_model(
    datasets: dict[int, WindowDataset],
    trees: int,
) -> ExtraTreesRegressor:
    features = np.vstack([dataset.features for dataset in datasets.values()])
    target = np.vstack(
        [dataset.target_heading_velocity for dataset in datasets.values()]
    )
    model = ExtraTreesRegressor(**model_parameters(trees))
    model.fit(features, target)
    return model


def learned_propagation(
    run: SynchronizedRun,
    calibration,
    initial_imu_rotation: np.ndarray,
    update_indices: np.ndarray,
    predicted_heading_velocity: np.ndarray,
    velocity_variance: np.ndarray,
    covariance_inflation: float,
    *,
    gravity_magnitude: float | None = None,
) -> tuple[Trajectory, np.ndarray, np.ndarray]:
    if gravity_magnitude is None:
        estimator = make_filter(run, calibration, initial_imu_rotation)
    else:
        estimator = make_filter(
            run,
            calibration,
            initial_imu_rotation,
            gravity_magnitude=gravity_magnitude,
        )
    estimator.measurement_noise_diag = np.maximum(
        velocity_variance * covariance_inflation,
        1.0e-4,
    )
    estimator.P[3:6, 3:6] = np.eye(3) * 0.25
    estimator.P[6:9, 6:9] = np.eye(3) * 0.01

    initial_body_rotation = run.rotation[0].as_matrix()
    body_from_imu = initial_body_rotation.T @ initial_imu_rotation
    prediction_by_index = {
        int(index): value
        for index, value in zip(update_indices, predicted_heading_velocity)
    }

    imu_rotations = [estimator.Rot.copy()]
    body_rotations = [estimator.Rot @ body_from_imu.T]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    position_std = [np.sqrt(np.clip(np.diag(estimator.P)[6:9], 0.0, None))]
    attitude_std = [
        np.rad2deg(np.sqrt(np.clip(np.diag(estimator.P)[0:3], 0.0, None)))
    ]
    learned_world_velocity = np.full((run.time.size, 3), np.nan)
    innovations = np.full((run.time.size, 3), np.nan)

    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        substeps = max(1, int(np.ceil(dt / 0.02)))
        for _ in range(substeps):
            estimator.predict(run.imu[index], dt / substeps)

        if index in prediction_by_index:
            body_rotation = estimator.Rot @ body_from_imu.T
            yaw = Rotation.from_matrix(body_rotation).as_euler("xyz")[2]
            cosine = float(np.cos(yaw))
            sine = float(np.sin(yaw))
            heading = prediction_by_index[index]
            world_velocity = np.array(
                [
                    cosine * heading[0] - sine * heading[1],
                    sine * heading[0] + cosine * heading[1],
                    heading[2],
                ]
            )
            innovation = world_velocity - estimator.v
            estimator.velocity_update(world_velocity)
            learned_world_velocity[index] = world_velocity
            innovations[index] = innovation

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

    return (
        Trajectory(
            time=run.time.copy(),
            imu_rotation=np.asarray(imu_rotations),
            body_rotation=np.asarray(body_rotations),
            velocity=np.asarray(velocities),
            position=np.asarray(positions),
            position_std=np.asarray(position_std),
            attitude_std_deg=np.asarray(attitude_std),
        ),
        learned_world_velocity,
        innovations,
    )


def plot_results(
    output: Path,
    run: SynchronizedRun,
    baseline: Trajectory,
    learned: Trajectory,
    learned_measurement: np.ndarray,
    baseline_position_error: np.ndarray,
    learned_position_error: np.ndarray,
    baseline_attitude_error: np.ndarray,
    learned_attitude_error: np.ndarray,
    cross_validation: dict[int, dict[str, object]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def draw_trajectory(
        axis: plt.Axes,
        cutoff: int,
        *,
        gt_full: bool = False,
    ) -> None:
        gt_cutoff = run.time.size if gt_full else cutoff
        axis.plot(
            run.position[:gt_cutoff, 0],
            run.position[:gt_cutoff, 1],
            "k",
            lw=1.4,
            label="GT",
        )
        axis.plot(
            baseline.position[:cutoff, 0],
            baseline.position[:cutoff, 1],
            color="0.60",
            lw=1.1,
            label="fixed-bias IMU DR",
        )
        axis.plot(
            learned.position[:cutoff, 0],
            learned.position[:cutoff, 1],
            color="tab:blue",
            lw=1.2,
            label="learned velocity + InEKF",
        )
        axis.scatter(run.position[0, 0], run.position[0, 1], c="green", s=40)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")

    twenty_seconds = int(np.searchsorted(run.time, 20.0, side="right"))
    draw_trajectory(axes[0], twenty_seconds)
    axes[0].set_title(
        "First 20 seconds\n"
        f"fixed {baseline_position_error[twenty_seconds - 1]:.2f} m, "
        f"learned {learned_position_error[twenty_seconds - 1]:.2f} m"
    )
    axes[0].legend()

    draw_trajectory(axes[1], run.time.size, gt_full=True)
    margin_x = max(1.00, 0.55 * np.ptp(run.position[:, 0]))
    margin_y = max(0.80, 0.55 * np.ptp(run.position[:, 1]))
    axes[1].set_xlim(
        np.min(run.position[:, 0]) - margin_x,
        np.max(run.position[:, 0]) + margin_x,
    )
    axes[1].set_ylim(
        np.min(run.position[:, 1]) - margin_y,
        np.max(run.position[:, 1]) + margin_y,
    )
    axes[1].set_title("Full trajectory near the GT range")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "01_trajectory_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for axis, horizon in zip(axes.ravel(), (30, 60, 120, run.time[-1])):
        cutoff = int(np.searchsorted(run.time, horizon, side="right"))
        axis.plot(
            run.position[:, 0],
            run.position[:, 1],
            "k",
            lw=1.2,
            alpha=0.7,
            label="GT full",
        )
        axis.plot(
            learned.position[:cutoff, 0],
            learned.position[:cutoff, 1],
            color="tab:blue",
            lw=1.2,
            label=f"learned + InEKF, 0–{min(horizon, run.time[-1]):.0f}s",
        )
        axis.scatter(run.position[0, 0], run.position[0, 1], c="green", s=30)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "05_horizon_trajectories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].semilogy(
        run.time,
        np.maximum(baseline_position_error, 1.0e-5),
        color="0.55",
        label="fixed-bias IMU DR",
    )
    axes[0].semilogy(
        run.time,
        np.maximum(learned_position_error, 1.0e-5),
        color="tab:blue",
        label="learned velocity + InEKF",
    )
    axes[0].set_ylabel("position error [m]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].plot(
        run.time,
        baseline_attitude_error,
        color="0.55",
        label="fixed-bias IMU DR",
    )
    axes[1].plot(
        run.time,
        learned_attitude_error,
        color="tab:blue",
        label="learned velocity + InEKF",
    )
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "02_error_comparison.png", dpi=180)
    plt.close(fig)

    gt_rpy = np.rad2deg(
        np.unwrap(run.rotation.as_euler("xyz"), axis=0)
    )
    baseline_rpy = np.rad2deg(
        np.unwrap(Rotation.from_matrix(baseline.body_rotation).as_euler("xyz"), axis=0)
    )
    learned_rpy = np.rad2deg(
        np.unwrap(Rotation.from_matrix(learned.body_rotation).as_euler("xyz"), axis=0)
    )
    baseline_rpy_error = (baseline_rpy - gt_rpy + 180.0) % 360.0 - 180.0
    learned_rpy_error = (learned_rpy - gt_rpy + 180.0) % 360.0 - 180.0

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 11),
        sharex="col",
        constrained_layout=True,
    )
    rpy_labels = ("roll", "pitch", "yaw")
    for axis_index, label in enumerate(rpy_labels):
        axes[axis_index, 0].plot(
            run.time,
            gt_rpy[:, axis_index],
            "k",
            lw=1.3,
            label="GT",
        )
        axes[axis_index, 0].plot(
            run.time,
            baseline_rpy[:, axis_index],
            color="0.60",
            lw=1.0,
            label="fixed-bias IMU DR",
        )
        axes[axis_index, 0].plot(
            run.time,
            learned_rpy[:, axis_index],
            color="tab:blue",
            lw=1.0,
            label="learned velocity + InEKF",
        )
        axes[axis_index, 0].set_ylabel(f"{label} [deg]")
        axes[axis_index, 0].grid(True, alpha=0.3)

        baseline_rmse = float(
            np.sqrt(np.mean(np.square(baseline_rpy_error[:, axis_index])))
        )
        learned_rmse = float(
            np.sqrt(np.mean(np.square(learned_rpy_error[:, axis_index])))
        )
        axes[axis_index, 1].plot(
            run.time,
            baseline_rpy_error[:, axis_index],
            color="0.60",
            lw=1.0,
            label=f"fixed-bias RMSE {baseline_rmse:.2f}°",
        )
        axes[axis_index, 1].plot(
            run.time,
            learned_rpy_error[:, axis_index],
            color="tab:blue",
            lw=1.0,
            label=f"learned RMSE {learned_rmse:.2f}°",
        )
        axes[axis_index, 1].axhline(0.0, color="k", lw=0.7, alpha=0.5)
        axes[axis_index, 1].set_ylabel(f"{label} error [deg]")
        axes[axis_index, 1].grid(True, alpha=0.3)
        axes[axis_index, 1].legend(loc="best")

    axes[0, 0].set_title("RPY estimates")
    axes[0, 1].set_title("Wrapped error relative to GT")
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.savefig(figures / "06_rpy_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ("vx", "vy", "vz")
    valid = np.all(np.isfinite(learned_measurement), axis=1)
    gt_quality = training_quality_mask(run)
    gt_velocity_for_plot = run.velocity.copy()
    gt_velocity_for_plot[~gt_quality] = np.nan
    for axis_index, axis in enumerate(axes):
        axis.plot(
            run.time,
            gt_velocity_for_plot[:, axis_index],
            "k",
            lw=1.0,
            label="GT evaluation only",
        )
        axis.plot(
            run.time,
            learned.velocity[:, axis_index],
            color="tab:blue",
            lw=0.9,
            label="InEKF velocity",
        )
        axis.plot(
            run.time[valid],
            learned_measurement[valid, axis_index],
            lw=0.6,
            alpha=0.6,
            color="tab:orange",
            label="learned IMU-only measurement",
        )
        axis.set_ylabel(f"{labels[axis_index]} [m/s]")
        axis.grid(True, alpha=0.3)
        displayed = np.r_[
            gt_velocity_for_plot[gt_quality, axis_index],
            learned.velocity[:, axis_index],
            learned_measurement[valid, axis_index],
        ]
        lower, upper = np.percentile(displayed[np.isfinite(displayed)], [0.5, 99.5])
        margin = max(0.2, 0.15 * (upper - lower))
        axis.set_ylim(lower - margin, upper + margin)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figures / "03_velocity_comparison.png", dpi=180)
    plt.close(fig)

    runs = sorted(cross_validation)
    errors = np.asarray(
        [cross_validation[run_id]["rmse_m_s"] for run_id in runs]
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(runs))
    width = 0.24
    for component in range(3):
        axis.bar(
            x + (component - 1) * width,
            errors[:, component],
            width,
            label=("vx", "vy", "vz")[component],
        )
    axis.set_xticks(x, [str(run_id) for run_id in runs])
    axis.set_xlabel("held-out training run")
    axis.set_ylabel("velocity RMSE [m/s]")
    axis.set_title("Leave-one-training-run-out cross-validation")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "04_training_cross_validation.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, object]) -> None:
    baseline = summary["metrics"]["fixed_bias_imu_only"]
    learned = summary["metrics"]["learned_velocity_inekf"]
    lines = [
        "# CF231 learned IMU-only velocity correction: held-out run 5",
        "",
        "## Leakage boundary",
        "",
        f"- Training runs: {summary['configuration']['training_runs']}",
        f"- Held-out test run: {summary['configuration']['test_run']}",
        "- Run-5 GT is used only after inference for metrics and plots.",
        "- Inference inputs: run-5 IMU windows and one initial state.",
        "- No run-5 GT/PWM/position-derived velocity is used by the model or filter.",
        "",
        "## Result",
        "",
        (
            f"- Fixed-bias IMU-only final position error: "
            f"{baseline['position_final_m']:.6g} m"
        ),
        (
            f"- Learned velocity + InEKF final position error: "
            f"{learned['position_final_m']:.6g} m"
        ),
        (
            f"- Fixed-bias position RMSE: {baseline['position_rmse_m']:.6g} m"
        ),
        (
            f"- Learned velocity + InEKF position RMSE: "
            f"{learned['position_rmse_m']:.6g} m"
        ),
        "",
        "## Interpretation",
        "",
        "- The learned output is a velocity measurement, not a position update.",
        "- The method is IMU-only at inference but data-driven, not universal.",
        "- Generalization must be tested on motions outside the CF231 figure-eight family.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_runs = parse_run_ids(args.training_runs)
    bias_runs = parse_run_ids(args.bias_runs)
    if args.test_run in training_runs or args.test_run in bias_runs:
        raise ValueError("The held-out test run cannot appear in training or bias runs.")
    if len(training_runs) < 3:
        raise ValueError("At least three training runs are required.")

    print("Loading training runs and constructing IMU windows...")
    synchronized_training = {
        run_id: synchronize_to_imu(load_cf231_run(dataset_path, run_id))
        for run_id in training_runs
    }
    datasets = {
        run_id: training_windows(
            run,
            args.window_samples,
            args.bins,
            args.update_stride,
        )
        for run_id, run in synchronized_training.items()
    }
    print("Cross-validating on training runs only...")
    cross_validation, velocity_variance, cross_validation_bias = cross_validate(
        datasets,
        args.trees,
    )
    velocity_variance = np.maximum(
        velocity_variance * args.velocity_noise_scale,
        1.0e-4,
    )
    print("Training final model without run 5...")
    model = train_model(datasets, args.trees)

    # The held-out run is now loaded for inference.  No target or quality mask
    # is constructed before the filter trajectory has been completed.
    print("Running held-out run-5 inference from IMU windows...")
    held_out = synchronize_to_imu(load_cf231_run(dataset_path, args.test_run))
    held_out_features, update_indices = inference_windows(
        held_out,
        args.window_samples,
        args.bins,
        1,
    )
    predicted_heading_velocity = (
        model.predict(held_out_features) - cross_validation_bias
    )

    calibration = estimate_sensor_calibration(
        dataset_path,
        bias_runs,
        args.static_seconds,
    )
    initial_rotation, initial_alignment = gravity_aligned_imu_rotation(
        held_out,
        calibration.accel_bias,
        args.static_seconds,
    )
    baseline = propagate(held_out, calibration, initial_rotation)
    learned, learned_measurement, innovations = learned_propagation(
        held_out,
        calibration,
        initial_rotation,
        update_indices,
        predicted_heading_velocity,
        velocity_variance,
        covariance_inflation=float(args.update_stride),
    )

    print("Scoring with run-5 GT after inference...")
    baseline_metrics, baseline_position_error, baseline_attitude_error = (
        calculate_metrics(baseline, held_out)
    )
    learned_metrics, learned_position_error, learned_attitude_error = (
        calculate_metrics(learned, held_out)
    )
    training_samples = {
        run_id: int(dataset.features.shape[0])
        for run_id, dataset in datasets.items()
    }
    summary = {
        "configuration": {
            "dataset": str(dataset_path),
            "training_runs": training_runs,
            "bias_runs": bias_runs,
            "test_run": args.test_run,
            "window_samples": args.window_samples,
            "window_seconds_approx": float(
                args.window_samples * calibration.sample_period_s
            ),
            "feature_bins": args.bins,
            "update_stride_samples": args.update_stride,
            "training_window_sampling_rate_hz_approx": float(
                1.0 / (args.update_stride * calibration.sample_period_s)
            ),
            "inference_model_prediction_rate_hz_approx": float(
                1.0 / calibration.sample_period_s
            ),
            "inekf_velocity_update_rate_hz_approx": float(
                1.0 / calibration.sample_period_s
            ),
            "inference_window_stride_samples": 1,
            "velocity_interpolation": "none; every window ends at the current IMU sample",
            "measurement_covariance_inflation": float(args.update_stride),
            "model": "ExtraTreesRegressor",
            "trees": args.trees,
            "training_samples": training_samples,
            "inference_inputs": [
                "held-out run IMU",
                "one initial yaw/position/velocity state",
            ],
            "uses_test_gt_during_training": False,
            "uses_test_gt_during_inference": False,
            "uses_test_pwm": False,
        },
        "cross_validation": {
            "per_run": cross_validation,
            "pooled_velocity_variance_m2_s2": velocity_variance.tolist(),
            "pooled_prediction_bias_m_s": cross_validation_bias.tolist(),
            "bias_correction_source": "training-run cross-validation only",
        },
        "fixed_bias": {
            "gyro_bias_rad_s": calibration.gyro_bias.tolist(),
            "accel_bias_m_s2": calibration.accel_bias.tolist(),
        },
        "initial_alignment": initial_alignment,
        "metrics": {
            "fixed_bias_imu_only": baseline_metrics,
            "learned_velocity_inekf": learned_metrics,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "training_runs": training_runs,
            "window_samples": args.window_samples,
            "bins": args.bins,
            "update_stride": args.update_stride,
            "target_frame": "yaw-heading",
        },
        output / "learned_velocity_model.joblib",
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=held_out.time,
        gt_position=held_out.position,
        gt_body_rotation=held_out.rotation.as_matrix(),
        baseline_position=baseline.position,
        baseline_velocity=baseline.velocity,
        baseline_body_rotation=baseline.body_rotation,
        learned_position=learned.position,
        learned_velocity=learned.velocity,
        learned_body_rotation=learned.body_rotation,
        learned_velocity_measurement=learned_measurement,
        velocity_innovation=innovations,
        update_indices=update_indices,
        predicted_heading_velocity=predicted_heading_velocity,
    )
    plot_results(
        output,
        held_out,
        baseline,
        learned,
        learned_measurement,
        baseline_position_error,
        learned_position_error,
        baseline_attitude_error,
        learned_attitude_error,
        cross_validation,
    )
    write_report(output / "report.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
