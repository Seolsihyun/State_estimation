"""Shallow learned IMU odometry with fixed-bias InEKF attitude on CF231.

Runs 3/4/9/10 supervise a small ExtraTrees velocity regressor and Run 5 is
strictly held out.  At inference, Run 5 contributes raw IMU windows only after
the initial state and one-time fixed calibration.  The InEKF propagates
attitude independently; learned heading-frame velocity is rotated by that
attitude and integrated for translation.  Keeping the two branches loose
prevents a biased learned velocity pseudo-measurement from corrupting attitude.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.cf231_loader import load_cf231_run, synchronize_to_imu
from validation.run_cf231_learned_velocity_leave5 import (
    cross_validate,
    inference_windows,
    parse_run_ids,
    train_model,
    training_quality_mask,
    training_windows,
)
from validation.run_cf231_pure_imu_improved import (
    detect_initial_stationary_window,
    estimate_gyro_intrinsic_matrix,
    estimate_session_fixed_bias,
    evaluate_method,
)
from validation.run_cf231_sensor_frame_leave5 import (
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/cf231_leave_one_out/csv",
    )
    parser.add_argument("--training-runs", default="3,4,9,10")
    parser.add_argument("--bias-runs", default="3,9,10")
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--training-stride", type=int, default=10)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "validation/results/cf231_leave5_shallow_learned_velocity_inekf",
    )
    return parser.parse_args()


def fixed_bias_attitude(test, calibration_runs, base_calibration, static_seconds):
    gyro_matrix, body_from_imu, gyro_fit = estimate_gyro_intrinsic_matrix(
        calibration_runs,
        static_seconds,
    )
    stationary_seconds, detector = detect_initial_stationary_window(test)
    gyro_bias, radial_accel_bias, session_info = estimate_session_fixed_bias(
        test,
        stationary_seconds,
    )
    static = test.time <= stationary_seconds
    accel_mean = np.mean(test.imu[static, 0:3], axis=0)
    expected_force = (
        body_from_imu.T
        @ test.rotation[0].as_matrix().T
        @ np.array([0.0, 0.0, 9.80665])
    )
    accel_bias = accel_mean - expected_force
    initial_imu_rotation = test.rotation[0].as_matrix() @ body_from_imu
    result = evaluate_method(
        "pure IMU full fixed calibration",
        test,
        base_calibration,
        gyro_bias,
        accel_bias,
        gyro_matrix,
        "previous",
        stationary_seconds,
        initial_rotation=initial_imu_rotation,
    )
    details = {
        "stationary_detector": detector,
        "gyro_bias_rad_s": gyro_bias.tolist(),
        "accel_bias_m_s2": accel_bias.tolist(),
        "radial_accel_bias_not_applied_m_s2": radial_accel_bias.tolist(),
        "gyro_intrinsic_matrix": gyro_matrix.tolist(),
        "gyro_fit": gyro_fit,
        "session_info": session_info,
    }
    return result, details


def integrate_learned_velocity(
    run,
    pure: Trajectory,
    update_indices: np.ndarray,
    predicted_heading_velocity: np.ndarray,
) -> tuple[Trajectory, np.ndarray]:
    yaw = Rotation.from_matrix(pure.body_rotation).as_euler("xyz")[:, 2]
    angle = yaw[update_indices]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    world_measurement = np.column_stack(
        (
            cosine * predicted_heading_velocity[:, 0]
            - sine * predicted_heading_velocity[:, 1],
            sine * predicted_heading_velocity[:, 0]
            + cosine * predicted_heading_velocity[:, 1],
            predicted_heading_velocity[:, 2],
        )
    )

    velocity = pure.velocity.copy()
    velocity[update_indices] = world_measurement
    position = np.empty_like(run.position)
    position[0] = run.position[0]
    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt

    zeros = np.zeros_like(position)
    return (
        Trajectory(
            time=run.time.copy(),
            imu_rotation=pure.imu_rotation.copy(),
            body_rotation=pure.body_rotation.copy(),
            velocity=velocity,
            position=position,
            position_std=zeros,
            attitude_std_deg=zeros,
        ),
        world_measurement,
    )


def load_legacy() -> Trajectory:
    data = np.load(
        ROOT / "validation/results/cf231_leave5_learned_velocity/trajectories.npz"
    )
    zeros = np.zeros_like(data["learned_position"])
    return Trajectory(
        time=data["time"],
        imu_rotation=data["learned_body_rotation"],
        body_rotation=data["learned_body_rotation"],
        velocity=data["learned_velocity"],
        position=data["learned_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )


def plot_results(output: Path, run, pure, legacy, shallow) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = {
        "Fixed-bias pure IMU": (pure, "0.65"),
        "Coupled learned velocity + InEKF": (legacy, "tab:orange"),
        "Shallow velocity + fixed-bias InEKF attitude": (shallow, "tab:blue"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for panel, (axis, horizon) in enumerate(
        zip(axes, (20.0, 60.0, float(run.time[-1])))
    ):
        mask = run.time <= horizon
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2, label="GT")
        visible = methods if panel == 0 else {
            label: value
            for label, value in methods.items()
            if label != "Fixed-bias pure IMU"
        }
        for label, (trajectory, color) in visible.items():
            axis.plot(
                trajectory.position[mask, 0],
                trajectory.position[mask, 1],
                color=color,
                lw=1.2,
                label=label,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend(fontsize=7)
    axes[0].set_title("0–20 s: pure IMU divergence")
    axes[1].set_title("0–60 s: learned methods")
    axes[2].set_title("Full Run 5: learned methods")
    fig.tight_layout()
    fig.savefig(figures / "01_trajectory_comparison.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, (trajectory, color) in methods.items():
        metrics, position_error, attitude_error = calculate_metrics(trajectory, run)
        axes[0].semilogy(
            run.time,
            np.maximum(position_error, 1.0e-5),
            color=color,
            label=f"{label} (RMSE {metrics['position_rmse_m']:.2f} m)",
        )
        axes[1].plot(run.time, attitude_error, color=color, label=label)
    axes[0].set_ylabel("3D position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "02_error_comparison.png", dpi=190)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    estimate_rpy = Rotation.from_matrix(shallow.body_rotation).as_euler(
        "xyz", degrees=True
    )
    rpy_error = (estimate_rpy - gt_rpy + 180.0) % 360.0 - 180.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    velocity_quality = training_quality_mask(run)
    gt_velocity = run.velocity.copy()
    gt_velocity[~velocity_quality] = np.nan
    for axis_index, name in enumerate(("x", "y", "z")):
        axes[0].plot(
            run.time,
            shallow.velocity[:, axis_index],
            label=f"estimated {name}",
        )
        axes[0].plot(
            run.time,
            gt_velocity[:, axis_index],
            "--",
            alpha=0.55,
            label=f"GT {name}",
        )
    robust_velocity = np.r_[
        shallow.velocity[velocity_quality].ravel(),
        run.velocity[velocity_quality].ravel(),
    ]
    lower, upper = np.quantile(robust_velocity, (0.005, 0.995))
    margin = max(0.1, 0.08 * float(upper - lower))
    axes[0].set_ylim(float(lower - margin), float(upper + margin))
    for axis_index, name in enumerate(("roll", "pitch", "yaw")):
        axes[1].plot(run.time, rpy_error[:, axis_index], label=name)
    axes[0].set_ylabel("velocity [m/s]")
    axes[1].set_ylabel("RPY error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "03_velocity_and_rpy.png", dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_ids = parse_run_ids(args.training_runs)
    bias_ids = parse_run_ids(args.bias_runs)
    if args.test_run in training_ids or args.test_run in bias_ids:
        raise ValueError("Held-out test run cannot be used for training or bias fit.")

    training_runs = {
        run_id: synchronize_to_imu(load_cf231_run(dataset, run_id))
        for run_id in training_ids
    }
    datasets = {
        run_id: training_windows(
            run,
            args.window_samples,
            args.bins,
            args.training_stride,
        )
        for run_id, run in training_runs.items()
    }
    cross_validation, velocity_variance, prediction_bias = cross_validate(
        datasets,
        args.trees,
    )
    model = train_model(datasets, args.trees)

    test = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    base_calibration = estimate_sensor_calibration(
        dataset,
        bias_ids,
        args.static_seconds,
    )
    pure_result, fixed_calibration = fixed_bias_attitude(
        test,
        [training_runs[run_id] for run_id in bias_ids],
        base_calibration,
        args.static_seconds,
    )
    held_out_features, update_indices = inference_windows(
        test,
        args.window_samples,
        args.bins,
        1,
    )
    predicted_heading_velocity = (
        model.predict(held_out_features) - prediction_bias
    )
    shallow, learned_world_velocity = integrate_learned_velocity(
        test,
        pure_result.trajectory,
        update_indices,
        predicted_heading_velocity,
    )
    legacy = load_legacy()

    pure_metrics, _, _ = calculate_metrics(pure_result.trajectory, test)
    legacy_metrics, _, _ = calculate_metrics(legacy, test)
    shallow_metrics, _, _ = calculate_metrics(shallow, test)
    plot_results(output, test, pure_result.trajectory, legacy, shallow)

    summary = {
        "configuration": {
            "dataset": str(dataset),
            "training_runs": training_ids,
            "bias_runs": bias_ids,
            "test_run": args.test_run,
            "model": "ExtraTreesRegressor (non-neural, non-deep)",
            "trees": args.trees,
            "window_samples": args.window_samples,
            "window_seconds": float(
                args.window_samples * base_calibration.sample_period_s
            ),
            "feature_bins": args.bins,
            "feature_dimensions": int(datasets[training_ids[0]].features.shape[1]),
            "training_stride": args.training_stride,
            "run5_gt_used_for_training": False,
            "run5_gt_used_during_inference": "initial state only; GT is loaded afterward for scoring",
            "coupling": "InEKF attitude and learned translation are propagated separately",
        },
        "training_samples": {
            str(run_id): int(dataset_.features.shape[0])
            for run_id, dataset_ in datasets.items()
        },
        "cross_validation": {
            "per_run": cross_validation,
            "pooled_prediction_bias_m_s": prediction_bias.tolist(),
            "pooled_velocity_variance_m2_s2": velocity_variance.tolist(),
        },
        "fixed_calibration": fixed_calibration,
        "metrics": {
            "pure_imu_full_fixed": pure_metrics,
            "legacy_coupled_learned_velocity_inekf": legacy_metrics,
            "shallow_velocity_fixed_attitude": shallow_metrics,
        },
        "interpretation": {
            "is_deep_learning": False,
            "run5_model_input": "IMU only after initial state",
            "training_label": "GT-derived heading-frame velocity from runs 3/4/9/10",
            "important_limitation": "A small residual velocity bias still integrates into long-horizon position drift; the method does not reproduce the full repeated figure-eight accurately.",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": model,
            "training_runs": training_ids,
            "window_samples": args.window_samples,
            "bins": args.bins,
            "prediction_bias_m_s": prediction_bias,
            "velocity_variance_m2_s2": velocity_variance,
        },
        output / "shallow_velocity_model.joblib",
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=test.time,
        gt_position=test.position,
        gt_rotation=test.rotation.as_matrix(),
        pure_position=pure_result.trajectory.position,
        pure_rotation=pure_result.trajectory.body_rotation,
        legacy_position=legacy.position,
        legacy_rotation=legacy.body_rotation,
        shallow_position=shallow.position,
        shallow_velocity=shallow.velocity,
        shallow_rotation=shallow.body_rotation,
        learned_world_velocity=learned_world_velocity,
        predicted_heading_velocity=predicted_heading_velocity,
        update_indices=update_indices,
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
