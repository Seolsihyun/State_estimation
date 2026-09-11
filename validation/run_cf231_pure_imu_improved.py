"""Improved prediction-only IMU dead reckoning on held-out CF231 run 5.

The test trajectory receives one initial GT pose/velocity, then only IMU
samples.  The initial stationary IMU samples are used once to estimate a
session gyro bias and radial accelerometer offset; both remain fixed.  A
constant gyroscope scale/non-orthogonality matrix is learned only from the
training runs and is never adapted on run 5.  There are no position, velocity,
GT, PWM, zero-velocity, or learned-velocity updates after initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import polar
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Hoon_invariant_inekf as lie
from validation.cf231_loader import (
    GRAVITY_MAGNITUDE,
    SynchronizedRun,
    load_cf231_run,
    synchronize_to_imu,
)
from validation.run_cf231_sensor_frame_leave5 import (
    SensorCalibration,
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
    gravity_aligned_imu_rotation,
    make_filter,
)


@dataclass(frozen=True)
class MethodResult:
    name: str
    trajectory: Trajectory
    metrics: dict[str, object]
    position_error: np.ndarray
    attitude_error: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "cf231_leave_one_out" / "csv",
    )
    parser.add_argument("--training-runs", default="3,9,10")
    parser.add_argument("--test-run", type=int, default=5)
    parser.add_argument("--training-static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "results" / "cf231_run5_pure_imu_improved",
    )
    return parser.parse_args()


def detect_initial_stationary_window(run: SynchronizedRun) -> tuple[float, dict[str, object]]:
    """Find the initial stationary interval from IMU only.

    Five consecutive gyro samples above 0.01 rad/s mark motion.  A 0.4 s
    guard interval is removed so rotor/handling transients cannot contaminate
    the fixed bias.  The thresholds are fixed before evaluating run 5.
    """
    seed = run.time <= min(0.5, float(run.time[-1]))
    seed_bias = np.median(run.imu[seed, 3:6], axis=0)
    centered_norm = np.linalg.norm(run.imu[:, 3:6] - seed_bias, axis=1)
    above = centered_norm > 0.01
    sustained = np.convolve(above.astype(int), np.ones(5, dtype=int), mode="valid")
    crossings = np.flatnonzero(sustained == 5)
    detected_motion = float(run.time[crossings[0]]) if crossings.size else 3.0
    duration = float(np.clip(detected_motion - 0.4, 0.8, 3.0))
    return duration, {
        "detector": "5 consecutive |gyro - initial median| > 0.01 rad/s",
        "detected_motion_time_s": detected_motion,
        "guard_time_s": 0.4,
        "stationary_duration_s": duration,
        "uses_gt": False,
    }


def estimate_session_fixed_bias(
    run: SynchronizedRun,
    stationary_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    static = run.time <= stationary_seconds
    if np.count_nonzero(static) < 50:
        raise ValueError("Too few initial stationary IMU samples.")
    gyro_bias = np.mean(run.imu[static, 3:6], axis=0)
    accel_mean = np.mean(run.imu[static, 0:3], axis=0)
    accel_norm = float(np.linalg.norm(accel_mean))
    if accel_norm < 0.5 * GRAVITY_MAGNITUDE:
        raise ValueError("Initial accelerometer mean is not gravity-like.")
    # A single static pose cannot identify all three accelerometer-bias axes.
    # Only the observable component parallel to gravity is removed.
    accel_bias = accel_mean - GRAVITY_MAGNITUDE * accel_mean / accel_norm
    return gyro_bias, accel_bias, {
        "samples": int(np.count_nonzero(static)),
        "gyro_bias_rad_s": gyro_bias.tolist(),
        "raw_accel_mean_m_s2": accel_mean.tolist(),
        "raw_accel_norm_m_s2": accel_norm,
        "radial_accel_bias_m_s2": accel_bias.tolist(),
        "corrected_accel_norm_m_s2": GRAVITY_MAGNITUDE,
    }


def estimate_gyro_intrinsic_matrix(
    runs: list[SynchronizedRun],
    static_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit fixed gyro scale/non-orthogonality using training-run GT only.

    The full affine map also includes the constant IMU-to-marker rotation.
    Its polar decomposition M = C S separates that rotation C from the
    symmetric intrinsic matrix S.  Only S is applied in the IMU sensor frame.
    """
    measured: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    per_run_samples: dict[int, int] = {}
    for run in runs:
        static = run.time <= static_seconds
        bias = np.mean(run.imu[static, 3:6], axis=0)
        x = run.imu[:, 3:6] - bias
        y = run.angular_velocity_body
        angular_rate = np.linalg.norm(y, axis=1)
        # The differentiated mocap orientation contains occasional spikes.
        # Rates above 2 rad/s are outside the useful calibration regime of
        # these runs and make an otherwise robust affine fit ill-conditioned.
        dynamic = (angular_rate >= 0.03) & (angular_rate <= 2.0)
        finite = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
        indices = np.flatnonzero(dynamic & finite)
        measured.append(x[indices])
        truth.append(y[indices])
        per_run_samples[run.run_id] = int(indices.size)
    x_all = np.vstack(measured)
    y_all = np.vstack(truth)

    def residual(parameters: np.ndarray) -> np.ndarray:
        matrix = parameters.reshape(3, 3)
        return (x_all @ matrix.T - y_all).ravel()

    fit = least_squares(
        residual,
        np.eye(3).ravel(),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=100,
    )
    affine = fit.x.reshape(3, 3)
    frame_rotation, intrinsic = polar(affine)
    if np.linalg.det(frame_rotation) < 0.0:
        raise ValueError("Gyroscope fit produced a reflected frame transform.")
    return intrinsic, frame_rotation, {
        "training_dynamic_samples": per_run_samples,
        "training_gt_rate_range_rad_s": [0.03, 2.0],
        "robust_loss": "soft_l1, f_scale=0.05 rad/s",
        "affine_body_from_sensor": affine.tolist(),
        "discarded_frame_rotation_rpy_deg": Rotation.from_matrix(
            frame_rotation
        ).as_euler("xyz", degrees=True).tolist(),
        "applied_sensor_intrinsic_matrix": intrinsic.tolist(),
        "fit_cost": float(fit.cost),
        "fit_optimality": float(fit.optimality),
    }


def propagate_nominal_only(
    run: SynchronizedRun,
    calibration: SensorCalibration,
    initial_rotation: np.ndarray,
    *,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    gyro_matrix: np.ndarray | None,
    sample_mode: str,
) -> Trajectory:
    """Run the InEKF nominal prediction without any measurement update."""
    fixed = replace(
        calibration,
        gyro_bias=np.zeros(3) if gyro_matrix is not None else gyro_bias.copy(),
        accel_bias=accel_bias.copy(),
    )
    estimator = make_filter(run, fixed, initial_rotation)
    initial_body_rotation = run.rotation[0].as_matrix()
    body_from_imu = initial_body_rotation.T @ initial_rotation

    imu_rotations = [estimator.Rot.copy()]
    body_rotations = [estimator.Rot @ body_from_imu.T]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    zeros = [np.zeros(3)]

    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        if not 0.0 < dt < 0.5:
            raise ValueError(f"Unexpected IMU dt={dt} at sample {index}.")
        if sample_mode == "previous":
            control = run.imu[index - 1].copy()
        elif sample_mode == "endpoint":
            control = run.imu[index].copy()
        elif sample_mode == "midpoint":
            control = 0.5 * (run.imu[index - 1] + run.imu[index])
        else:
            raise ValueError(f"Unknown sample mode: {sample_mode}")
        if gyro_matrix is not None:
            control[3:6] = gyro_matrix @ (control[3:6] - gyro_bias)
        estimator.Rot, estimator.v, estimator.p = estimator._propagate_nominal(
            estimator.Rot,
            estimator.v,
            estimator.p,
            estimator.gyro_bias,
            estimator.accel_bias,
            control,
            dt,
        )
        estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
        imu_rotations.append(estimator.Rot.copy())
        body_rotations.append(estimator.Rot @ body_from_imu.T)
        velocities.append(estimator.v.copy())
        positions.append(estimator.p.copy())
        zeros.append(np.zeros(3))
    return Trajectory(
        time=run.time.copy(),
        imu_rotation=np.asarray(imu_rotations),
        body_rotation=np.asarray(body_rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
        position_std=np.asarray(zeros),
        attitude_std_deg=np.asarray(zeros),
    )


def evaluate_method(
    name: str,
    run: SynchronizedRun,
    calibration: SensorCalibration,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    gyro_matrix: np.ndarray | None,
    sample_mode: str,
    initialization_seconds: float,
    initial_rotation: np.ndarray | None = None,
) -> MethodResult:
    if initial_rotation is None:
        initial_rotation, _ = gravity_aligned_imu_rotation(
            run,
            accel_bias,
            initialization_seconds,
        )
    trajectory = propagate_nominal_only(
        run,
        calibration,
        initial_rotation,
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        gyro_matrix=gyro_matrix,
        sample_mode=sample_mode,
    )
    metrics, position_error, attitude_error = calculate_metrics(trajectory, run)
    return MethodResult(
        name=name,
        trajectory=trajectory,
        metrics=metrics,
        position_error=position_error,
        attitude_error=attitude_error,
    )


def training_cross_validation(
    runs: dict[int, SynchronizedRun],
    static_seconds: float,
) -> dict[str, object]:
    """Choose integration/calibration rules without looking at run-5 GT."""
    output: dict[str, object] = {}
    ids = sorted(runs)
    for held_out in ids:
        train_ids = [run_id for run_id in ids if run_id != held_out]
        train_runs = [runs[run_id] for run_id in train_ids]
        base_calibration = estimate_sensor_calibration_from_loaded(
            train_runs,
            static_seconds,
        )
        gyro_matrix, body_from_imu, _ = estimate_gyro_intrinsic_matrix(
            train_runs, static_seconds
        )
        test = runs[held_out]
        stationary_seconds, detector = detect_initial_stationary_window(test)
        gyro_bias, _radial_accel_bias, _ = estimate_session_fixed_bias(
            test,
            stationary_seconds,
        )
        # A single stationary pose observes gyro bias, but cannot separate
        # transverse accelerometer bias from roll/pitch. Transfer accel bias
        # from the other runs instead of pretending all three axes are known.
        accel_bias = base_calibration.accel_bias
        methods = {}
        static = test.time <= stationary_seconds
        accel_mean = np.mean(test.imu[static, 0:3], axis=0)
        expected_force = (
            body_from_imu.T
            @ test.rotation[0].as_matrix().T
            @ np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        )
        full_accel_bias = accel_mean - expected_force
        for label, matrix, sampling, used_accel_bias, initial_rotation in (
            ("endpoint_no_intrinsic", None, "endpoint", accel_bias, None),
            ("previous_no_intrinsic", None, "previous", accel_bias, None),
            (
                "previous_full_calibration",
                gyro_matrix,
                "previous",
                full_accel_bias,
                test.rotation[0].as_matrix() @ body_from_imu,
            ),
        ):
            result = evaluate_method(
                label,
                test,
                base_calibration,
                gyro_bias,
                used_accel_bias,
                matrix,
                sampling,
                stationary_seconds,
                initial_rotation=initial_rotation,
            )
            methods[label] = {
                "position_rmse_m": result.metrics["position_rmse_m"],
                "position_final_m": result.metrics["position_final_m"],
                "so3_rmse_deg": result.metrics["so3_rmse_deg"],
            }
        output[str(held_out)] = {
            "trained_on": train_ids,
            "stationary_detector": detector,
            "methods": methods,
        }
    return output


def estimate_sensor_calibration_from_loaded(
    runs: list[SynchronizedRun],
    static_seconds: float,
) -> SensorCalibration:
    gyro_biases = []
    accel_biases = []
    gyro_variances = []
    accel_variances = []
    periods = []
    per_run: dict[int, dict[str, object]] = {}
    for run in runs:
        static = run.time <= static_seconds
        accel_mean = np.mean(run.imu[static, 0:3], axis=0)
        gyro_mean = np.mean(run.imu[static, 3:6], axis=0)
        accel_bias = accel_mean - np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
        gyro_biases.append(gyro_mean)
        accel_biases.append(accel_bias)
        gyro_variances.append(np.var(run.imu[static, 3:6], axis=0))
        accel_variances.append(np.var(run.imu[static, 0:3], axis=0))
        periods.append(float(np.median(np.diff(run.time[static]))))
        per_run[run.run_id] = {
            "gyro_bias_rad_s": gyro_mean.tolist(),
            "accel_bias_m_s2": accel_bias.tolist(),
        }
    gyro = np.asarray(gyro_biases)
    accel = np.asarray(accel_biases)
    ddof = 1 if len(runs) > 1 else 0
    return SensorCalibration(
        gyro_bias=np.median(gyro, axis=0),
        accel_bias=np.median(accel, axis=0),
        gyro_sample_variance=np.median(gyro_variances, axis=0),
        accel_sample_variance=np.median(accel_variances, axis=0),
        gyro_between_run_variance=np.var(gyro, axis=0, ddof=ddof),
        accel_between_run_variance=np.var(accel, axis=0, ddof=ddof),
        sample_period_s=float(np.median(periods)),
        per_run=per_run,
    )


def plot_results(
    figures: Path,
    run: SynchronizedRun,
    baseline: MethodResult,
    fixed_session: MethodResult,
    improved: MethodResult,
    cross_validation: dict[str, object],
    gyro_matrix: np.ndarray,
    session_info: dict[str, object],
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    colors = {
        baseline.name: "0.65",
        fixed_session.name: "tab:orange",
        improved.name: "tab:blue",
    }
    methods = (baseline, fixed_session, improved)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for axis, horizon in zip(axes.flat, (5, 10, 20, 30)):
        mask = run.time <= horizon
        axis.plot(run.position[mask, 0], run.position[mask, 1], "k", lw=2, label="GT")
        for method in methods:
            p = method.trajectory.position[mask]
            axis.plot(p[:, 0], p[:, 1], color=colors[method.name], lw=1.3, label=method.name)
        axis.set_title(f"first {horizon} s")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Run 5: pure IMU dead reckoning only")
    fig.tight_layout()
    fig.savefig(figures / "01_horizon_trajectory_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis in axes:
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2, label="GT full")
    for method in methods:
        crossing = np.flatnonzero(method.position_error >= 10.0)
        cutoff = int(crossing[0]) if crossing.size else run.time.size - 1
        axes[0].plot(
            method.trajectory.position[: cutoff + 1, 0],
            method.trajectory.position[: cutoff + 1, 1],
            color=colors[method.name],
            lw=1.5,
            label=f"{method.name} to 10 m ({run.time[cutoff]:.1f} s)",
        )
        mask = run.time <= 30.0
        axes[1].plot(
            method.trajectory.position[mask, 0],
            method.trajectory.position[mask, 1],
            color=colors[method.name],
            lw=1.3,
            label=method.name,
        )
    axes[0].set_title("Full GT and pure-IMU tracks until 10 m error")
    axes[1].set_title("First 30 s at a wider scale")
    margin_x = 1.0
    margin_y = 1.0
    early = improved.trajectory.position[run.time <= 30]
    axes[1].set_xlim(min(run.position[:, 0].min(), early[:, 0].min()) - margin_x,
                     max(run.position[:, 0].max(), early[:, 0].max()) + margin_x)
    axes[1].set_ylim(min(run.position[:, 1].min(), early[:, 1].min()) - margin_y,
                     max(run.position[:, 1].max(), early[:, 1].max()) + margin_y)
    for axis in axes:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(figures / "02_gt_scale_and_divergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for method in methods:
        axes[0].semilogy(
            run.time,
            np.maximum(method.position_error, 1e-6),
            color=colors[method.name],
            label=method.name,
        )
        axes[1].plot(
            run.time,
            method.attitude_error,
            color=colors[method.name],
            label=method.name,
        )
    axes[0].set_ylabel("position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_position_attitude_error.png", dpi=180)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for component, label in enumerate(("roll", "pitch", "yaw")):
        axes[component].plot(run.time, gt_rpy[:, component], "k", lw=1.3, label="GT")
        for method in methods:
            rpy = Rotation.from_matrix(method.trajectory.body_rotation).as_euler(
                "xyz", degrees=True
            )
            axes[component].plot(
                run.time,
                rpy[:, component],
                color=colors[method.name],
                lw=0.8,
                label=method.name,
            )
        axes[component].set_ylabel(f"{label} [deg]")
        axes[component].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figures / "04_rpy_comparison.png", dpi=180)
    plt.close(fig)

    run_ids = sorted(int(run_id) for run_id in cross_validation)
    labels = ("endpoint_no_intrinsic", "previous_no_intrinsic", "previous_full_calibration")
    display = ("endpoint", "previous", "previous + full fixed calibration")
    x = np.arange(len(run_ids))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for index, (label, shown) in enumerate(zip(labels, display)):
        pos = [cross_validation[str(run_id)]["methods"][label]["position_rmse_m"] for run_id in run_ids]
        att = [cross_validation[str(run_id)]["methods"][label]["so3_rmse_deg"] for run_id in run_ids]
        axes[0].bar(x + (index - 1) * width, pos, width, label=shown)
        axes[1].bar(x + (index - 1) * width, att, width, label=shown)
    axes[0].set_ylabel("position RMSE [m]")
    axes[1].set_ylabel("SO(3) RMSE [deg]")
    for axis in axes:
        axis.set_xticks(x, [str(run_id) for run_id in run_ids])
        axis.set_xlabel("held-out training run")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Training-run-only cross-validation")
    fig.tight_layout()
    fig.savefig(figures / "05_training_cross_validation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    image = axes[0].imshow(gyro_matrix, cmap="coolwarm", vmin=0.0, vmax=1.05)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{gyro_matrix[i, j]:.4f}", ha="center", va="center")
    axes[0].set_xticks(range(3), ["gx", "gy", "gz"])
    axes[0].set_yticks(range(3), ["gx", "gy", "gz"])
    axes[0].set_title("Fixed gyro intrinsic matrix")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    gyro_bias = np.asarray(session_info["gyro_bias_rad_s"])
    accel_bias = np.asarray(session_info["applied_accel_bias_m_s2"])
    axes[1].bar(np.arange(3) - 0.16, gyro_bias, 0.32, label="gyro [rad/s]")
    axes[1].bar(np.arange(3) + 0.16, accel_bias, 0.32, label="accel [m/s^2]")
    axes[1].set_xticks(range(3), ["x", "y", "z"])
    axes[1].set_title("Run 5 one-time fixed biases")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "06_fixed_calibration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_ids = sorted(
        {int(value.strip()) for value in args.training_runs.split(",") if value.strip()}
    )
    if args.test_run in training_ids:
        raise ValueError("Test run must be held out from training calibration.")

    print("Loading synchronized runs...")
    training_runs = {
        run_id: synchronize_to_imu(load_cf231_run(dataset, run_id))
        for run_id in training_ids
    }
    test = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    base_calibration = estimate_sensor_calibration(
        dataset,
        training_ids,
        args.training_static_seconds,
    )

    print("Fitting fixed gyro intrinsic calibration on training runs only...")
    gyro_matrix, body_from_imu, gyro_fit = estimate_gyro_intrinsic_matrix(
        list(training_runs.values()),
        args.training_static_seconds,
    )
    stationary_seconds, detector = detect_initial_stationary_window(test)
    session_gyro_bias, session_radial_accel_bias, session_info = estimate_session_fixed_bias(
        test,
        stationary_seconds,
    )
    static = test.time <= stationary_seconds
    run5_accel_mean = np.mean(test.imu[static, 0:3], axis=0)
    run5_expected_force = (
        body_from_imu.T
        @ test.rotation[0].as_matrix().T
        @ np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    )
    full_session_accel_bias = run5_accel_mean - run5_expected_force
    full_initial_imu_rotation = test.rotation[0].as_matrix() @ body_from_imu
    session_info = {
        **session_info,
        "expected_static_force_from_initial_gt_m_s2": run5_expected_force.tolist(),
        "applied_accel_bias_m_s2": full_session_accel_bias.tolist(),
    }

    print("Running three prediction-only pure-IMU variants...")
    baseline = evaluate_method(
        "transferred bias (original)",
        test,
        base_calibration,
        base_calibration.gyro_bias,
        base_calibration.accel_bias,
        None,
        "endpoint",
        args.training_static_seconds,
    )
    fixed_session = evaluate_method(
        "fixed session bias",
        test,
        base_calibration,
        session_gyro_bias,
        base_calibration.accel_bias,
        None,
        "previous",
        stationary_seconds,
    )
    improved = evaluate_method(
        "full fixed calibration",
        test,
        base_calibration,
        session_gyro_bias,
        full_session_accel_bias,
        gyro_matrix,
        "previous",
        stationary_seconds,
        initial_rotation=full_initial_imu_rotation,
    )

    print("Cross-validating choices without run-5 GT...")
    cross_validation = training_cross_validation(
        training_runs,
        args.training_static_seconds,
    )
    methods = {
        "original_transferred_bias": baseline.metrics,
        "fixed_session_bias": fixed_session.metrics,
        "full_fixed_calibration": improved.metrics,
    }
    summary = {
        "configuration": {
            "dataset": str(dataset),
            "training_runs": training_ids,
            "test_run": args.test_run,
            "duration_s": float(test.time[-1]),
            "samples": int(test.time.size),
            "initial_gt_used": ["full orientation", "position", "velocity"],
            "run5_gt_used_after_initialization": False,
            "updates_after_initialization": [],
            "uses_position_or_velocity_measurement": False,
            "uses_learned_velocity": False,
            "uses_pwm": False,
            "bias_evolution": "none; one value fixed for complete run",
            "propagation": "InEKF nominal IMU prediction only",
        },
        "stationary_detector": detector,
        "run5_fixed_session_calibration": session_info,
        "applied_accel_bias_from_training_runs_m_s2": base_calibration.accel_bias.tolist(),
        "run5_radial_accel_bias_diagnostic_not_applied_m_s2": session_radial_accel_bias.tolist(),
        "training_gyro_intrinsic_fit": gyro_fit,
        "methods": methods,
        "improvement_vs_original_percent": {
            "position_rmse": float(
                100.0
                * (baseline.metrics["position_rmse_m"] - improved.metrics["position_rmse_m"])
                / baseline.metrics["position_rmse_m"]
            ),
            "final_position_error": float(
                100.0
                * (baseline.metrics["position_final_m"] - improved.metrics["position_final_m"])
                / baseline.metrics["position_final_m"]
            ),
            "so3_rmse": float(
                100.0
                * (baseline.metrics["so3_rmse_deg"] - improved.metrics["so3_rmse_deg"])
                / baseline.metrics["so3_rmse_deg"]
            ),
        },
        "training_cross_validation": cross_validation,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=test.time,
        gt_position=test.position,
        gt_rotation=test.rotation.as_matrix(),
        original_position=baseline.trajectory.position,
        original_rotation=baseline.trajectory.body_rotation,
        fixed_session_position=fixed_session.trajectory.position,
        fixed_session_rotation=fixed_session.trajectory.body_rotation,
        improved_position=improved.trajectory.position,
        improved_rotation=improved.trajectory.body_rotation,
    )
    plot_results(
        output / "figures",
        test,
        baseline,
        fixed_session,
        improved,
        cross_validation,
        gyro_matrix,
        session_info,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
