"""Stagewise CF231 IMU-only learned dead-reckoning validation.

The held-out protocol is fixed throughout:

* Runs 3/4/9/10 supervise the motion network.
* Run 5 supplies only its initial state, initial IMU calibration interval, and
  subsequent IMU samples during inference.
* Run-5 ground truth is read only after each trajectory has been generated.

The script compares four stages:

1. fixed-bias prediction-only IMU dead reckoning;
2. the existing Small-TCN velocity branch with loose velocity integration;
3. an AirIO-inspired body-frame velocity and uncertainty virtual sensor that
   performs an actual InEKF update;
4. the same virtual sensor trained with auxiliary multi-horizon displacement
   consistency, inspired by TLIO's relative-displacement supervision.

No PWM, thrust, GNSS, camera, bounded-motion, or position pseudo-measurement is
used at Run-5 inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import hat_so3, plus_right
from utils.filter_math import kalman_update
from validation.cf231_loader import SynchronizedRun, load_cf231_run, synchronize_to_imu
from validation.run_cf231_learned_velocity_leave5 import (
    heading_velocity,
    parse_run_ids,
    training_quality_mask,
)
from validation.run_cf231_pure_imu_improved import (
    detect_initial_stationary_window,
    estimate_gyro_intrinsic_matrix,
    estimate_session_fixed_bias,
)
from validation.run_cf231_sensor_frame_leave5 import (
    SensorCalibration,
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
    make_filter,
)
from validation.run_cf231_shallow_learned_velocity_inekf import fixed_bias_attitude


@dataclass(frozen=True)
class MotionSequences:
    inputs: np.ndarray
    velocity: np.ndarray
    displacement: np.ndarray
    end_indices: np.ndarray


@dataclass(frozen=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    displacement_mean: np.ndarray
    displacement_std: np.ndarray


@dataclass(frozen=True)
class Prediction:
    body_velocity: np.ndarray
    body_sigma: np.ndarray
    body_displacement: np.ndarray
    end_indices: np.ndarray


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: MotionSequences,
        stats: Normalization,
        augment: bool,
    ) -> None:
        self.sequences = sequences
        self.stats = stats
        self.augment = augment

    def __len__(self) -> int:
        return self.sequences.inputs.shape[0]

    def __getitem__(self, index: int):
        inputs = self.sequences.inputs[index].copy()
        if self.augment:
            # One constant offset per window plus small white noise. The
            # magnitude is deliberately below the observed between-run spread.
            inputs[:, 0:3] += np.random.normal(0.0, 0.01, (1, 3))
            inputs[:, 3:6] += np.random.normal(0.0, 1.5e-4, (1, 3))
            inputs[:, 0:3] += np.random.normal(0.0, 0.006, inputs[:, 0:3].shape)
            inputs[:, 3:6] += np.random.normal(0.0, 8.0e-5, inputs[:, 3:6].shape)
        inputs = (inputs - self.stats.input_mean) / self.stats.input_std
        velocity = (
            self.sequences.velocity[index] - self.stats.velocity_mean
        ) / self.stats.velocity_std
        displacement = (
            self.sequences.displacement[index] - self.stats.displacement_mean
        ) / self.stats.displacement_std
        return (
            torch.from_numpy(inputs.T.astype(np.float32)),
            torch.from_numpy(velocity.astype(np.float32)),
            torch.from_numpy(displacement.astype(np.float32)),
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class AirIOLiteTCN(nn.Module):
    """Compact body-velocity virtual sensor with heteroscedastic uncertainty."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(9, channels, 7, padding=3),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            TemporalBlock(channels, 1),
            TemporalBlock(channels, 2),
            TemporalBlock(channels, 4),
        )
        self.shared = nn.Sequential(
            nn.Linear(2 * channels, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
        )
        self.velocity_head = nn.Linear(64, 3)
        self.log_sigma_head = nn.Linear(64, 3)
        self.displacement_head = nn.Linear(64, 9)

    def forward(self, value):
        feature = self.encoder(value)
        pooled = torch.cat((feature.mean(dim=-1), feature[:, :, -1]), dim=1)
        shared = self.shared(pooled)
        velocity = self.velocity_head(shared)
        log_sigma = torch.clamp(self.log_sigma_head(shared), -4.0, 2.0)
        displacement = self.displacement_head(shared).reshape(-1, 3, 3)
        return velocity, log_sigma, displacement


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
    parser.add_argument("--validation-run", type=int, default=10)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--training-stride", type=int, default=5)
    parser.add_argument("--update-stride", type=int, default=10)
    parser.add_argument("--validation-epochs", type=int, default=14)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--displacement-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--reuse-trained",
        action="store_true",
        help="Reuse saved stage models and only rebuild filters, metrics, and figures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/results/cf231_imu_only_learned_inekf_stages",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def session_fixed_imu(
    run: SynchronizedRun,
    gyro_matrix: np.ndarray,
    body_from_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply one fixed per-session calibration estimated at initialization."""
    stationary_seconds, _ = detect_initial_stationary_window(run)
    gyro_bias, _radial_accel_bias, _ = estimate_session_fixed_bias(
        run, stationary_seconds
    )
    static = run.time <= stationary_seconds
    initial_imu_rotation = run.rotation[0].as_matrix() @ body_from_imu
    expected_force = initial_imu_rotation.T @ np.array([0.0, 0.0, 9.80665])
    accel_bias = np.mean(run.imu[static, 0:3], axis=0) - expected_force
    corrected = run.imu.copy()
    corrected[:, 0:3] -= accel_bias
    corrected[:, 3:6] = (gyro_matrix @ (corrected[:, 3:6] - gyro_bias).T).T
    return corrected, gyro_bias, accel_bias, stationary_seconds


def imu_rotvec(rotation_matrices: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation_matrices).as_rotvec()


def body_velocity(run: SynchronizedRun, imu_rotation: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", imu_rotation, run.velocity)


def make_sequences(
    run: SynchronizedRun,
    imu: np.ndarray,
    imu_rotation: np.ndarray,
    window: int,
    downsample: int,
    stride: int,
    require_quality: bool,
) -> MotionSequences:
    velocity = body_velocity(run, imu_rotation)
    attitude = imu_rotvec(imu_rotation)
    features = np.column_stack((imu, attitude))
    horizons = (50, 100, 200)
    quality = training_quality_mask(run) if require_quality else np.ones(run.time.size, bool)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    indices: list[int] = []
    for end in range(window - 1, run.time.size, stride):
        start = end - window + 1
        if require_quality and not (
            quality[end]
            and np.mean(quality[start : end + 1]) >= 0.90
            and np.all(np.isfinite(velocity[end]))
        ):
            continue
        rotation_end = imu_rotation[end]
        displacement_targets = []
        for horizon in horizons:
            horizon_start = max(0, end - horizon + 1)
            delta_world = run.position[end] - run.position[horizon_start]
            displacement_targets.append(rotation_end.T @ delta_world)
        inputs.append(features[start : end + 1 : downsample])
        targets.append(velocity[end])
        displacements.append(np.asarray(displacement_targets))
        indices.append(end)
    return MotionSequences(
        inputs=np.asarray(inputs, dtype=np.float32),
        velocity=np.asarray(targets, dtype=np.float32),
        displacement=np.asarray(displacements, dtype=np.float32),
        end_indices=np.asarray(indices, dtype=np.int64),
    )


def make_inference_sequences(
    imu: np.ndarray,
    imu_rotation: np.ndarray,
    window: int,
    downsample: int,
) -> MotionSequences:
    """Build network inputs without touching test-run position or velocity."""
    features = np.column_stack((imu, imu_rotvec(imu_rotation)))
    indices = np.arange(window - 1, features.shape[0], dtype=np.int64)
    inputs = np.asarray(
        [features[end - window + 1 : end + 1 : downsample] for end in indices],
        dtype=np.float32,
    )
    return MotionSequences(
        inputs=inputs,
        velocity=np.zeros((indices.size, 3), dtype=np.float32),
        displacement=np.zeros((indices.size, 3, 3), dtype=np.float32),
        end_indices=indices,
    )


def concatenate_sequences(items: list[MotionSequences]) -> MotionSequences:
    return MotionSequences(
        inputs=np.concatenate([item.inputs for item in items], axis=0),
        velocity=np.concatenate([item.velocity for item in items], axis=0),
        displacement=np.concatenate([item.displacement for item in items], axis=0),
        end_indices=np.concatenate([item.end_indices for item in items], axis=0),
    )


def fit_normalization(items: list[MotionSequences]) -> Normalization:
    merged = concatenate_sequences(items)
    return Normalization(
        input_mean=merged.inputs.mean(axis=(0, 1)),
        input_std=np.maximum(merged.inputs.std(axis=(0, 1)), 1.0e-5),
        velocity_mean=merged.velocity.mean(axis=0),
        velocity_std=np.maximum(merged.velocity.std(axis=0), 1.0e-4),
        displacement_mean=merged.displacement.mean(axis=0),
        displacement_std=np.maximum(merged.displacement.std(axis=0), 1.0e-4),
    )


def fit_model(
    items: list[MotionSequences],
    channels: int,
    epochs: int,
    batch_size: int,
    displacement_weight: float,
    seed: int,
) -> tuple[AirIOLiteTCN, Normalization, list[dict[str, float]]]:
    set_seed(seed)
    stats = fit_normalization(items)
    dataset = SequenceDataset(concatenate_sequences(items), stats, augment=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = AirIOLiteTCN(channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for _epoch in range(epochs):
        totals = np.zeros(4, dtype=float)
        count = 0
        model.train()
        for inputs, target_velocity, target_displacement in loader:
            optimizer.zero_grad(set_to_none=True)
            mean, log_sigma, displacement = model(inputs)
            point = nn.functional.smooth_l1_loss(mean, target_velocity)
            normalized_residual = mean - target_velocity
            nll = 0.5 * torch.mean(
                torch.exp(-2.0 * log_sigma) * torch.square(normalized_residual)
                + 2.0 * log_sigma
            )
            mean_bias = torch.mean(
                torch.square(torch.mean(normalized_residual, dim=0))
            )
            displacement_loss = nn.functional.smooth_l1_loss(
                displacement, target_displacement
            )
            loss = (
                point
                + 0.05 * nll
                + 0.15 * mean_bias
                + displacement_weight * displacement_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = inputs.shape[0]
            totals += batch_count * np.array(
                [
                    float(loss.detach()),
                    float(point.detach()),
                    float(nll.detach()),
                    float(displacement_loss.detach()),
                ]
            )
            count += batch_count
        scheduler.step()
        values = totals / max(count, 1)
        history.append(
            {
                "total": float(values[0]),
                "velocity": float(values[1]),
                "uncertainty": float(values[2]),
                "displacement": float(values[3]),
            }
        )
    return model, stats, history


def predict_model(
    model: AirIOLiteTCN,
    stats: Normalization,
    sequences: MotionSequences,
    batch_size: int,
    sigma_scale: np.ndarray | None = None,
) -> Prediction:
    dataset = SequenceDataset(sequences, stats, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    means: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _velocity, _displacement in loader:
            mean, log_sigma, displacement = model(inputs)
            means.append(mean.cpu().numpy())
            sigmas.append(np.exp(log_sigma.cpu().numpy()))
            displacements.append(displacement.cpu().numpy())
    normalized_mean = np.vstack(means)
    normalized_sigma = np.vstack(sigmas)
    normalized_displacement = np.concatenate(displacements, axis=0)
    velocity = normalized_mean * stats.velocity_std + stats.velocity_mean
    sigma = normalized_sigma * stats.velocity_std
    if sigma_scale is not None:
        sigma *= np.asarray(sigma_scale, dtype=float).reshape(1, 3)
    sigma = np.clip(sigma, 0.02, 1.5)
    displacement = (
        normalized_displacement * stats.displacement_std
        + stats.displacement_mean
    )
    return Prediction(
        body_velocity=velocity,
        body_sigma=sigma,
        body_displacement=displacement,
        end_indices=sequences.end_indices.copy(),
    )


def calibrate_uncertainty(
    prediction: Prediction,
    target: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    residual = prediction.body_velocity - target
    denominator = np.maximum(prediction.body_sigma, 1.0e-4)
    scale = np.sqrt(np.mean(np.square(residual / denominator), axis=0))
    scale = np.clip(scale, 0.5, 10.0)
    calibrated = prediction.body_sigma * scale[None, :]
    return scale, {
        "velocity_rmse_m_s": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
        "velocity_bias_m_s": np.mean(residual, axis=0).tolist(),
        "raw_sigma_mean_m_s": np.mean(prediction.body_sigma, axis=0).tolist(),
        "sigma_scale": scale.tolist(),
        "calibrated_1sigma_coverage": np.mean(
            np.abs(residual) <= calibrated, axis=0
        ).tolist(),
        "calibrated_2sigma_coverage": np.mean(
            np.abs(residual) <= 2.0 * calibrated, axis=0
        ).tolist(),
    }


def body_velocity_update(
    estimator,
    measurement_body: np.ndarray,
    sigma_body: np.ndarray,
    mahalanobis_gate: float = 16.27,
) -> tuple[bool, float]:
    """Right-invariant error update for h(X)=R^T v in the IMU frame."""
    predicted_body = estimator.Rot.T @ estimator.v
    innovation = np.asarray(measurement_body, dtype=float) - predicted_body
    H = np.zeros((3, estimator.error_dim), dtype=float)
    H[:, 0:3] = hat_so3(predicted_body)
    H[:, 3:6] = np.eye(3)
    variance = np.square(np.asarray(sigma_body, dtype=float))
    Rm = np.diag(np.maximum(variance, 4.0e-4))
    S = H @ estimator.P @ H.T + Rm + 1.0e-12 * np.eye(3)
    mahalanobis = float(innovation.T @ np.linalg.solve(S, innovation))
    if not np.isfinite(mahalanobis) or mahalanobis > mahalanobis_gate:
        return False, mahalanobis
    _, P_update, estimator.innovation, estimator.S, estimator.K = kalman_update(
        np.zeros(estimator.error_dim), estimator.P, innovation, H, Rm
    )
    estimator.delta = estimator._bounded_delta(estimator.K @ innovation)
    estimator.Rot, estimator.v, estimator.p = lie.from_matrix(
        plus_right(
            lie.as_matrix(estimator.Rot, estimator.v, estimator.p),
            estimator.delta[:9],
        )
    )
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    estimator.P = estimator._stabilize_covariance(P_update)
    return True, mahalanobis


def world_velocity_update(
    estimator,
    measurement_world: np.ndarray,
    covariance_world: np.ndarray,
) -> None:
    """World-velocity update for the older heading-frame Small TCN baseline."""
    innovation = np.asarray(measurement_world, dtype=float) - estimator.v
    H = np.zeros((3, estimator.error_dim), dtype=float)
    # Under X Exp(delta), v_plus ~= v + R dv.
    H[:, 3:6] = estimator.Rot
    Rm = np.asarray(covariance_world, dtype=float).reshape(3, 3)
    _, P_update, estimator.innovation, estimator.S, estimator.K = kalman_update(
        np.zeros(estimator.error_dim), estimator.P, innovation, H, Rm
    )
    estimator.delta = estimator._bounded_delta(estimator.K @ innovation)
    estimator.Rot, estimator.v, estimator.p = lie.from_matrix(
        plus_right(
            lie.as_matrix(estimator.Rot, estimator.v, estimator.p),
            estimator.delta[:9],
        )
    )
    estimator.X = lie.as_matrix(estimator.Rot, estimator.v, estimator.p)
    estimator.P = estimator._stabilize_covariance(P_update)


def run_heading_velocity_inekf(
    run: SynchronizedRun,
    calibration: SensorCalibration,
    initial_imu_rotation: np.ndarray,
    body_from_imu: np.ndarray,
    gyro_matrix: np.ndarray,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    update_indices: np.ndarray,
    predicted_heading_velocity: np.ndarray,
    heading_sigma: np.ndarray,
    update_stride: int,
) -> Trajectory:
    """Fuse the existing heading-frame Small-TCN output as an InEKF update."""
    fixed = replace(
        calibration,
        gyro_bias=np.zeros(3),
        accel_bias=np.asarray(accel_bias, dtype=float).copy(),
    )
    estimator = make_filter(run, fixed, initial_imu_rotation)
    estimator.gyro_bias[:] = 0.0
    estimator.accel_bias = np.asarray(accel_bias, dtype=float).copy()
    estimator.update_biases = False
    estimator.process_noise_diag[0:3] = np.maximum(
        calibration.gyro_sample_variance, 1.0e-10
    )
    estimator.process_noise_diag[3:6] = np.maximum(
        calibration.accel_sample_variance, 1.0e-8
    )
    estimator.process_noise_diag[9:15] = 0.0
    estimator.P.fill(0.0)
    estimator.P[0:3, 0:3] = np.eye(3) * np.deg2rad(0.5) ** 2
    estimator.P[3:6, 3:6] = np.eye(3) * 0.05**2
    estimator.P[6:9, 6:9] = np.eye(3) * 1.0e-6
    lookup = {
        int(index): row
        for row, index in enumerate(update_indices)
        if int(index) % update_stride == 0
    }
    imu_rotations = [estimator.Rot.copy()]
    body_rotations = [estimator.Rot @ body_from_imu.T]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    position_std = [np.sqrt(np.maximum(np.diag(estimator.P)[6:9], 0.0))]
    attitude_std = [
        np.rad2deg(np.sqrt(np.maximum(np.diag(estimator.P)[0:3], 0.0)))
    ]
    heading_covariance = np.diag(np.square(heading_sigma))
    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        control = run.imu[index - 1].copy()
        control[3:6] = gyro_matrix @ (control[3:6] - gyro_bias)
        estimator.predict(control, dt)
        row = lookup.get(index)
        if row is not None:
            body_rotation = estimator.Rot @ body_from_imu.T
            yaw = Rotation.from_matrix(body_rotation).as_euler("xyz")[2]
            cosine, sine = np.cos(yaw), np.sin(yaw)
            heading_to_world = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            measurement_world = (
                heading_to_world @ predicted_heading_velocity[row]
            )
            covariance_world = (
                heading_to_world @ heading_covariance @ heading_to_world.T
            )
            world_velocity_update(estimator, measurement_world, covariance_world)
        imu_rotations.append(estimator.Rot.copy())
        body_rotations.append(estimator.Rot @ body_from_imu.T)
        velocities.append(estimator.v.copy())
        positions.append(estimator.p.copy())
        position_std.append(
            np.sqrt(np.maximum(np.diag(estimator.P)[6:9], 0.0))
        )
        attitude_std.append(
            np.rad2deg(np.sqrt(np.maximum(np.diag(estimator.P)[0:3], 0.0)))
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


def run_learned_inekf(
    run: SynchronizedRun,
    calibration: SensorCalibration,
    initial_imu_rotation: np.ndarray,
    body_from_imu: np.ndarray,
    gyro_matrix: np.ndarray,
    gyro_bias: np.ndarray,
    accel_bias: np.ndarray,
    prediction: Prediction,
    update_stride: int,
) -> tuple[Trajectory, dict[str, object]]:
    fixed = replace(
        calibration,
        gyro_bias=np.zeros(3),
        accel_bias=np.asarray(accel_bias, dtype=float).copy(),
    )
    estimator = make_filter(run, fixed, initial_imu_rotation)
    estimator.gyro_bias[:] = 0.0
    estimator.accel_bias = np.asarray(accel_bias, dtype=float).copy()
    estimator.update_biases = False
    estimator.process_noise_diag[0:3] = np.maximum(
        calibration.gyro_sample_variance, 1.0e-10
    )
    estimator.process_noise_diag[3:6] = np.maximum(
        calibration.accel_sample_variance, 1.0e-8
    )
    estimator.process_noise_diag[9:15] = 0.0
    estimator.P.fill(0.0)
    estimator.P[0:3, 0:3] = np.eye(3) * np.deg2rad(0.5) ** 2
    estimator.P[3:6, 3:6] = np.eye(3) * 0.05**2
    estimator.P[6:9, 6:9] = np.eye(3) * 1.0e-6

    lookup = {
        int(index): row
        for row, index in enumerate(prediction.end_indices)
        if int(index) % update_stride == 0
    }
    imu_rotations = [estimator.Rot.copy()]
    body_rotations = [estimator.Rot @ body_from_imu.T]
    velocities = [estimator.v.copy()]
    positions = [estimator.p.copy()]
    position_std = [np.sqrt(np.maximum(np.diag(estimator.P)[6:9], 0.0))]
    attitude_std = [
        np.rad2deg(np.sqrt(np.maximum(np.diag(estimator.P)[0:3], 0.0)))
    ]
    accepted = 0
    rejected = 0
    mahalanobis_values: list[float] = []
    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        if not 0.0 < dt < 0.5:
            raise ValueError(f"Unexpected IMU dt={dt} at sample {index}.")
        control = run.imu[index - 1].copy()
        control[3:6] = gyro_matrix @ (control[3:6] - gyro_bias)
        estimator.predict(control, dt)
        row = lookup.get(index)
        if row is not None:
            was_accepted, mahalanobis = body_velocity_update(
                estimator,
                prediction.body_velocity[row],
                prediction.body_sigma[row],
            )
            mahalanobis_values.append(mahalanobis)
            if was_accepted:
                accepted += 1
            else:
                rejected += 1
        imu_rotations.append(estimator.Rot.copy())
        body_rotations.append(estimator.Rot @ body_from_imu.T)
        velocities.append(estimator.v.copy())
        positions.append(estimator.p.copy())
        position_std.append(
            np.sqrt(np.maximum(np.diag(estimator.P)[6:9], 0.0))
        )
        attitude_std.append(
            np.rad2deg(np.sqrt(np.maximum(np.diag(estimator.P)[0:3], 0.0)))
        )
    trajectory = Trajectory(
        time=run.time.copy(),
        imu_rotation=np.asarray(imu_rotations),
        body_rotation=np.asarray(body_rotations),
        velocity=np.asarray(velocities),
        position=np.asarray(positions),
        position_std=np.asarray(position_std),
        attitude_std_deg=np.asarray(attitude_std),
    )
    diagnostics = {
        "candidate_updates": accepted + rejected,
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "acceptance_rate": accepted / max(accepted + rejected, 1),
        "mahalanobis_median": float(np.nanmedian(mahalanobis_values)),
        "mahalanobis_p95": float(np.nanpercentile(mahalanobis_values, 95)),
    }
    return trajectory, diagnostics


def load_existing_stages(run: SynchronizedRun) -> tuple[Trajectory, Trajectory]:
    data = np.load(
        ROOT
        / "validation/results/cf231_leave5_small_tcn_velocity_inekf/trajectories.npz"
    )
    zeros = np.zeros_like(data["pure_position"])
    pure = Trajectory(
        time=data["time"],
        imu_rotation=data["tcn_rotation"],
        body_rotation=data["tcn_rotation"],
        velocity=np.zeros_like(data["pure_position"]),
        position=data["pure_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )
    loose = Trajectory(
        time=data["time"],
        imu_rotation=data["tcn_rotation"],
        body_rotation=data["tcn_rotation"],
        velocity=data["tcn_velocity"],
        position=data["tcn_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )
    if pure.time.shape != run.time.shape or not np.allclose(pure.time, run.time):
        raise ValueError("Existing baseline does not match the current Run-5 timeline.")
    return pure, loose


def load_legacy_learned_velocity(run: SynchronizedRun) -> Trajectory:
    data = np.load(
        ROOT / "validation/results/cf231_leave5_learned_velocity/trajectories.npz"
    )
    zeros = np.zeros_like(data["learned_position"])
    trajectory = Trajectory(
        time=data["time"],
        imu_rotation=data["learned_body_rotation"],
        body_rotation=data["learned_body_rotation"],
        velocity=data["learned_velocity"],
        position=data["learned_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )
    if trajectory.time.shape != run.time.shape or not np.allclose(
        trajectory.time, run.time
    ):
        raise ValueError("Legacy learned-velocity trajectory timeline mismatch.")
    return trajectory


def smoothed_xy_for_plot(
    time: np.ndarray,
    position: np.ndarray,
    mask: np.ndarray | None = None,
    sigma_s: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed XY path for display only."""
    if mask is None:
        time_view = time
        xy = position[:, :2]
    else:
        time_view = time[mask]
        xy = position[mask, :2]
    if xy.shape[0] < 5:
        return xy[:, 0], xy[:, 1]

    dt = float(np.median(np.diff(time_view)))
    if not np.isfinite(dt) or dt <= 0.0:
        return xy[:, 0], xy[:, 1]
    sigma = sigma_s / dt
    if sigma < 1.0:
        return xy[:, 0], xy[:, 1]
    smooth = gaussian_filter1d(
        xy,
        sigma=sigma,
        axis=0,
        mode="nearest",
        truncate=3.0,
    )
    return smooth[:, 0], smooth[:, 1]


def plot_results(
    output: Path,
    run: SynchronizedRun,
    body_from_imu: np.ndarray,
    methods: dict[str, tuple[Trajectory, str]],
    predictions: dict[str, Prediction],
    metrics: dict[str, dict[str, object]],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    tcn_names = [
        name for name in ("small_tcn_loose", "small_tcn_heading_inekf")
        if name in methods
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panel_specs = (
        (20.0, 0.55, "First 20 s near GT"),
        (60.0, 1.45, "First 60 s, wider view"),
        (float(run.time[-1]), 2.70, "Full Run 5, widest view"),
    )
    for axis, (horizon, gt_margin, title) in zip(axes, panel_specs):
        mask = run.time <= horizon
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2.3, label="GT")
        for name in tcn_names:
            trajectory, color = methods[name]
            x_smooth, y_smooth = smoothed_xy_for_plot(
                trajectory.time, trajectory.position, mask
            )
            axis.plot(
                x_smooth,
                y_smooth,
                color=color,
                lw=1.1,
                label=name,
            )
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_xlim(
            run.position[:, 0].min() - gt_margin,
            run.position[:, 0].max() + gt_margin,
        )
        axis.set_ylim(
            run.position[:, 1].min() - gt_margin,
            run.position[:, 1].max() + gt_margin,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=7)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(figures / "01_learned_trajectory_stages.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, (trajectory, color) in methods.items():
        _method_metrics, position_error, attitude_error = calculate_metrics(
            trajectory, run
        )
        axes[0].semilogy(
            run.time,
            np.maximum(position_error, 1.0e-5),
            color=color,
            lw=1.1,
            label=f"{name}: {metrics[name]['position_rmse_m']:.2f} m",
        )
        axes[1].plot(run.time, attitude_error, color=color, lw=1.0, label=name)
    axes[0].set_ylabel("3D position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "02_all_stage_errors.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pure = methods["fixed_bias_imu"][0]
    mask = run.time <= 20.0
    pure_x, pure_y = smoothed_xy_for_plot(pure.time, pure.position, mask)
    axes[0].plot(run.position[:, 0], run.position[:, 1], "k", lw=2, label="GT")
    axes[0].plot(pure_x, pure_y, "0.55", lw=1.2,
                 label="fixed-bias IMU (first 20 s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Pure IMU divergence onset")
    axes[0].set_aspect("equal", adjustable="box")
    names = ["fixed_bias_imu", *tcn_names]
    x = np.arange(len(names))
    rmse = [metrics[name]["position_rmse_m"] for name in names]
    final = [metrics[name]["position_final_m"] for name in names]
    width = 0.36
    axes[1].bar(x - width / 2, rmse, width, label="RMSE")
    axes[1].bar(x + width / 2, final, width, label="final")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, names, rotation=18, ha="right")
    axes[1].set_ylabel("position error [m], log scale")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Stagewise error summary")
    fig.tight_layout()
    fig.savefig(figures / "03_pure_divergence_and_summary.png", dpi=190)
    plt.close(fig)

    gt_imu_rotation = run.rotation.as_matrix() @ body_from_imu
    gt_body_velocity = body_velocity(run, gt_imu_rotation)
    quality = training_quality_mask(run)
    gt_body_velocity = gt_body_velocity.copy()
    gt_body_velocity[~quality] = np.nan
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ("x", "y", "z")
    for method_name, prediction in predictions.items():
        color = methods[method_name][1]
        for axis_index, axis in enumerate(axes):
            index = prediction.end_indices
            axis.plot(
                run.time[index],
                prediction.body_velocity[:, axis_index],
                color=color,
                lw=0.8,
                label=method_name,
            )
            if axis_index == 0:
                axis.fill_between(
                    run.time[index],
                    prediction.body_velocity[:, axis_index]
                    - 2.0 * prediction.body_sigma[:, axis_index],
                    prediction.body_velocity[:, axis_index]
                    + 2.0 * prediction.body_sigma[:, axis_index],
                    color=color,
                    alpha=0.08,
                )
    for axis_index, axis in enumerate(axes):
        axis.plot(run.time, gt_body_velocity[:, axis_index], "k", lw=1.0, label="GT")
        axis.set_ylabel(f"v{labels[axis_index]} [m/s]")
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figures / "04_body_velocity_and_uncertainty.png", dpi=190)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for component, label in enumerate(("roll", "pitch", "yaw")):
        axes[component].plot(run.time, gt_rpy[:, component], "k", lw=1.2, label="GT")
        for name, (trajectory, color) in methods.items():
            estimated = Rotation.from_matrix(trajectory.body_rotation).as_euler(
                "xyz", degrees=True
            )
            axes[component].plot(
                run.time, estimated[:, component], color=color, lw=0.75, label=name
            )
        axes[component].set_ylabel(f"{label} [deg]")
        axes[component].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(figures / "05_rpy_all_stages.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(run.position[:, 0], run.position[:, 1], "k", lw=2.3, label="GT")
    for name in ("small_tcn_loose", "small_tcn_heading_inekf"):
        trajectory, color = methods[name]
        x_smooth, y_smooth = smoothed_xy_for_plot(
            trajectory.time, trajectory.position
        )
        axes[0].plot(
            x_smooth, y_smooth, color=color, lw=1.0, label=name,
        )
    margin = 0.35
    axes[0].set_xlim(run.position[:, 0].min() - margin, run.position[:, 0].max() + margin)
    axes[0].set_ylim(run.position[:, 1].min() - margin, run.position[:, 1].max() + margin)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Full Run 5 near the GT range")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    for name in ("small_tcn_loose", "small_tcn_heading_inekf"):
        trajectory, color = methods[name]
        _metric, position_error, _attitude_error = calculate_metrics(trajectory, run)
        axes[1].plot(
            run.time, position_error, color=color, lw=1.0,
            label=f"{name}: {metrics[name]['position_rmse_m']:.2f} m RMSE",
        )
    axes[1].set_title("Best learned methods")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("3D position error [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "06_best_methods_near_gt.png", dpi=190)
    plt.close(fig)

    comparison_names = (
        "fixed_bias_imu",
        "learned_velocity_inekf",
        "small_tcn_loose",
        "small_tcn_heading_inekf",
    )
    comparison_labels = {
        "fixed_bias_imu": "fixed-bias IMU DR",
        "learned_velocity_inekf": "learned velocity",
        "small_tcn_loose": "Small TCN",
        "small_tcn_heading_inekf": "Small TCN + InEKF",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panel_specs = (
        (20.0, None, "First 20 s: pure IMU divergence"),
        (60.0, "learned", "First 60 s: learned methods near GT"),
        (float(run.time[-1]), "learned", "Full Run 5: accumulated drift"),
    )
    for axis, (horizon, axis_mode, title) in zip(axes, panel_specs):
        mask = run.time <= horizon
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2.2, label="GT")
        for name in comparison_names:
            trajectory, color = methods[name]
            x_smooth, y_smooth = smoothed_xy_for_plot(
                trajectory.time, trajectory.position, mask
            )
            axis.plot(
                x_smooth,
                y_smooth,
                color=color,
                lw=1.15,
                label=comparison_labels[name],
            )
        if axis_mode is None:
            all_x = [run.position[:, 0]]
            all_y = [run.position[:, 1]]
            for name in comparison_names:
                trajectory = methods[name][0]
                all_x.append(trajectory.position[mask, 0])
                all_y.append(trajectory.position[mask, 1])
            x_min = min(float(np.nanmin(values)) for values in all_x)
            x_max = max(float(np.nanmax(values)) for values in all_x)
            y_min = min(float(np.nanmin(values)) for values in all_y)
            y_max = max(float(np.nanmax(values)) for values in all_y)
            x_margin = 0.12 * max(x_max - x_min, 1.0)
            y_margin = 0.12 * max(y_max - y_min, 1.0)
            axis.set_xlim(x_min - x_margin, x_max + x_margin)
            axis.set_ylim(y_min - y_margin, y_max + y_margin)
        elif axis_mode == "learned":
            all_x = [run.position[:, 0]]
            all_y = [run.position[:, 1]]
            for name in comparison_names:
                if name == "fixed_bias_imu":
                    continue
                trajectory = methods[name][0]
                all_x.append(trajectory.position[mask, 0])
                all_y.append(trajectory.position[mask, 1])
            x_min = min(float(np.nanmin(values)) for values in all_x)
            x_max = max(float(np.nanmax(values)) for values in all_x)
            y_min = min(float(np.nanmin(values)) for values in all_y)
            y_max = max(float(np.nanmax(values)) for values in all_y)
            x_margin = 0.12 * max(x_max - x_min, 1.0)
            y_margin = 0.12 * max(y_max - y_min, 1.0)
            axis.set_xlim(x_min - x_margin, x_max + x_margin)
            axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "07_fixed_vs_tcn_trajectory_comparison.png", dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_ids = parse_run_ids(args.training_runs)
    bias_ids = parse_run_ids(args.bias_runs)
    if args.test_run in training_ids or args.test_run in bias_ids:
        raise ValueError("Run 5 must be held out from training and bias fitting.")
    if args.validation_run not in training_ids:
        raise ValueError("Validation run must be one of the training runs.")

    runs = {
        run_id: synchronize_to_imu(load_cf231_run(dataset, run_id))
        for run_id in training_ids
    }
    calibration_runs = [runs[run_id] for run_id in bias_ids]
    base_calibration = estimate_sensor_calibration(
        dataset, bias_ids, args.static_seconds
    )
    gyro_matrix, body_from_imu, gyro_fit = estimate_gyro_intrinsic_matrix(
        calibration_runs, args.static_seconds
    )

    training_sequences: dict[int, MotionSequences] = {}
    session_training_calibration: dict[str, object] = {}
    for run_id, run in runs.items():
        corrected, gyro_bias, accel_bias, stationary_seconds = session_fixed_imu(
            run, gyro_matrix, body_from_imu
        )
        gt_imu_rotation = run.rotation.as_matrix() @ body_from_imu
        training_sequences[run_id] = make_sequences(
            run,
            corrected,
            gt_imu_rotation,
            args.window_samples,
            args.downsample,
            args.training_stride,
            True,
        )
        session_training_calibration[str(run_id)] = {
            "stationary_seconds": stationary_seconds,
            "gyro_bias_rad_s": gyro_bias.tolist(),
            "accel_bias_m_s2": accel_bias.tolist(),
            "samples": int(training_sequences[run_id].inputs.shape[0]),
        }

    train_without_validation = [
        training_sequences[run_id]
        for run_id in training_ids
        if run_id != args.validation_run
    ]
    validation_sequences = training_sequences[args.validation_run]
    validation_results: dict[str, object] = {}
    sigma_scales: dict[str, np.ndarray] = {}
    if args.reuse_trained:
        previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        validation_results = previous["validation_uncertainty_calibration"]
        sigma_scales = {
            name: np.asarray(details["sigma_scale"], dtype=float)
            for name, details in validation_results.items()
        }
    else:
        for name, displacement_weight in (
            ("airio_velocity_inekf", 0.0),
            ("airio_displacement_inekf", args.displacement_weight),
        ):
            model, stats, history = fit_model(
                train_without_validation,
                args.channels,
                args.validation_epochs,
                args.batch_size,
                displacement_weight,
                args.seed + (1 if displacement_weight > 0 else 0),
            )
            prediction = predict_model(
                model, stats, validation_sequences, args.batch_size
            )
            sigma_scale, calibration_details = calibrate_uncertainty(
                prediction, validation_sequences.velocity
            )
            sigma_scales[name] = sigma_scale
            validation_results[name] = {
                **calibration_details,
                "final_loss": history[-1],
            }
            print(f"validation {name}: {validation_results[name]}", flush=True)

    test = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    pure_result, fixed_calibration = fixed_bias_attitude(
        test, calibration_runs, base_calibration, args.static_seconds
    )
    corrected_test, test_gyro_bias, test_accel_bias, test_stationary = session_fixed_imu(
        test, gyro_matrix, body_from_imu
    )
    test_sequences = make_inference_sequences(
        corrected_test,
        pure_result.trajectory.imu_rotation,
        args.window_samples,
        args.downsample,
    )
    initial_imu_rotation = test.rotation[0].as_matrix() @ body_from_imu

    final_models: dict[str, AirIOLiteTCN] = {}
    final_stats: dict[str, Normalization] = {}
    final_histories: dict[str, list[dict[str, float]]] = {}
    learned_predictions: dict[str, Prediction] = {}
    learned_trajectories: dict[str, Trajectory] = {}
    filter_diagnostics: dict[str, object] = {}
    for name, displacement_weight in (
        ("airio_velocity_inekf", 0.0),
        ("airio_displacement_inekf", args.displacement_weight),
    ):
        if args.reuse_trained:
            checkpoint = torch.load(
                output / f"{name}.pt", map_location="cpu", weights_only=False
            )
            model = AirIOLiteTCN(args.channels)
            model.load_state_dict(checkpoint["state_dict"])
            stats = Normalization(**checkpoint["normalization"])
            history = [previous["final_training_loss"][name]]
        else:
            model, stats, history = fit_model(
                [training_sequences[run_id] for run_id in training_ids],
                args.channels,
                args.final_epochs,
                args.batch_size,
                displacement_weight,
                args.seed + (11 if displacement_weight > 0 else 10),
            )
        prediction = predict_model(
            model,
            stats,
            test_sequences,
            args.batch_size,
            sigma_scale=sigma_scales[name],
        )
        trajectory, diagnostics = run_learned_inekf(
            test,
            base_calibration,
            initial_imu_rotation,
            body_from_imu,
            gyro_matrix,
            test_gyro_bias,
            test_accel_bias,
            prediction,
            args.update_stride,
        )
        final_models[name] = model
        final_stats[name] = stats
        final_histories[name] = history
        learned_predictions[name] = prediction
        learned_trajectories[name] = trajectory
        filter_diagnostics[name] = diagnostics
        print(f"filter {name}: {diagnostics}", flush=True)

    existing_pure, existing_loose = load_existing_stages(test)
    legacy_learned_velocity = load_legacy_learned_velocity(test)
    legacy_data = np.load(
        ROOT
        / "validation/results/cf231_leave5_small_tcn_velocity_inekf/trajectories.npz"
    )
    legacy_summary = json.loads(
        (
            ROOT
            / "validation/results/cf231_leave5_small_tcn_velocity_inekf/summary.json"
        ).read_text(encoding="utf-8")
    )
    heading_sigma = np.asarray(
        legacy_summary["cross_validation"]["pooled_velocity_rmse_m_s"],
        dtype=float,
    )
    heading_inekf = run_heading_velocity_inekf(
        test,
        base_calibration,
        initial_imu_rotation,
        body_from_imu,
        gyro_matrix,
        test_gyro_bias,
        test_accel_bias,
        legacy_data["update_indices"],
        legacy_data["predicted_heading_velocity"],
        heading_sigma,
        args.update_stride,
    )
    methods = {
        "fixed_bias_imu": (existing_pure, "0.55"),
        "learned_velocity_inekf": (legacy_learned_velocity, "tab:blue"),
        "small_tcn_loose": (existing_loose, "tab:orange"),
        "small_tcn_heading_inekf": (heading_inekf, "tab:purple"),
        "airio_velocity_inekf": (
            learned_trajectories["airio_velocity_inekf"],
            "tab:blue",
        ),
        "airio_displacement_inekf": (
            learned_trajectories["airio_displacement_inekf"],
            "tab:green",
        ),
    }
    metrics = {
        name: calculate_metrics(trajectory, test)[0]
        for name, (trajectory, _color) in methods.items()
    }

    # Score learned velocity only after the complete Run-5 inference pass.
    gt_imu_rotation = test.rotation.as_matrix() @ body_from_imu
    gt_body_velocity = body_velocity(test, gt_imu_rotation)
    test_quality = training_quality_mask(test)
    test_velocity_metrics: dict[str, object] = {}
    for name, prediction in learned_predictions.items():
        target = gt_body_velocity[prediction.end_indices]
        valid = test_quality[prediction.end_indices]
        residual = prediction.body_velocity[valid] - target[valid]
        valid_sigma = prediction.body_sigma[valid]
        test_velocity_metrics[name] = {
            "rmse_m_s": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
            "bias_m_s": np.mean(residual, axis=0).tolist(),
            "one_sigma_coverage": np.mean(
                np.abs(residual) <= valid_sigma, axis=0
            ).tolist(),
            "two_sigma_coverage": np.mean(
                np.abs(residual) <= 2.0 * valid_sigma, axis=0
            ).tolist(),
        }
    heading_target = heading_velocity(test)
    heading_indices = legacy_data["update_indices"]
    heading_valid = test_quality[heading_indices]
    heading_residual = (
        legacy_data["predicted_heading_velocity"][heading_valid]
        - heading_target[heading_indices][heading_valid]
    )
    test_velocity_metrics["small_tcn_heading"] = {
        "rmse_m_s": np.sqrt(np.mean(np.square(heading_residual), axis=0)).tolist(),
        "bias_m_s": np.mean(heading_residual, axis=0).tolist(),
        "valid_samples": int(np.count_nonzero(heading_valid)),
    }

    plot_results(
        output, test, body_from_imu, methods, learned_predictions, metrics
    )
    parameter_count = int(
        sum(value.numel() for value in final_models["airio_velocity_inekf"].parameters())
    )
    summary = {
        "protocol": {
            "dataset": str(dataset),
            "training_runs": training_ids,
            "bias_runs": bias_ids,
            "validation_run_for_uncertainty": args.validation_run,
            "test_run": args.test_run,
            "test_runtime_inputs": "initial state, initial fixed IMU calibration, then Run-5 IMU only",
            "run5_gt_during_inference": False,
            "excluded_inputs": ["PWM", "thrust", "GNSS", "camera", "bounded-motion constraint"],
            "window_samples": args.window_samples,
            "window_seconds": float(
                args.window_samples * base_calibration.sample_period_s
            ),
            "update_stride": args.update_stride,
            "update_period_s": float(
                args.update_stride * base_calibration.sample_period_s
            ),
            "network_parameters": parameter_count,
            "fixed_bias_policy": "one value per run, estimated once from the initial IMU-only stationary detector and frozen",
        },
        "training_session_calibration": session_training_calibration,
        "test_fixed_calibration": {
            **fixed_calibration,
            "stationary_seconds": test_stationary,
            "gyro_bias_rad_s": test_gyro_bias.tolist(),
            "accel_bias_m_s2": test_accel_bias.tolist(),
            "gyro_intrinsic_fit": gyro_fit,
        },
        "validation_uncertainty_calibration": validation_results,
        "filter_diagnostics": filter_diagnostics,
        "metrics": metrics,
        "test_velocity_metrics_post_inference": test_velocity_metrics,
        "final_training_loss": {
            name: history[-1] for name, history in final_histories.items()
        },
        "method_definitions": {
            "fixed_bias_imu": "fixed-bias InEKF prediction only",
            "learned_velocity_inekf": "legacy ExtraTrees-style learned heading velocity fused through InEKF",
            "small_tcn_loose": "existing heading-velocity TCN; velocity replaced and integrated outside InEKF",
            "small_tcn_heading_inekf": "the same held-out heading-velocity prediction fused as an actual InEKF velocity update using training-run CV covariance",
            "airio_velocity_inekf": "body IMU + estimated attitude -> body velocity and uncertainty -> actual InEKF update",
            "airio_displacement_inekf": "same filter update, with auxiliary 0.5/1/2 s body displacement supervision during training",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for name, model in final_models.items():
        stats = final_stats[name]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "normalization": {
                    key: np.asarray(value)
                    for key, value in stats.__dict__.items()
                },
                "sigma_scale": sigma_scales[name],
                "protocol": summary["protocol"],
            },
            output / f"{name}.pt",
        )
    np.savez_compressed(
        output / "trajectories.npz",
        time=test.time,
        gt_position=test.position,
        gt_rotation=test.rotation.as_matrix(),
        fixed_bias_position=existing_pure.position,
        learned_velocity_position=legacy_learned_velocity.position,
        learned_velocity_rotation=legacy_learned_velocity.body_rotation,
        small_tcn_position=existing_loose.position,
        small_tcn_heading_inekf_position=heading_inekf.position,
        small_tcn_heading_inekf_rotation=heading_inekf.body_rotation,
        airio_velocity_position=learned_trajectories["airio_velocity_inekf"].position,
        airio_velocity_rotation=learned_trajectories["airio_velocity_inekf"].body_rotation,
        airio_displacement_position=learned_trajectories["airio_displacement_inekf"].position,
        airio_displacement_rotation=learned_trajectories["airio_displacement_inekf"].body_rotation,
        update_indices=test_sequences.end_indices,
        airio_velocity_prediction=learned_predictions["airio_velocity_inekf"].body_velocity,
        airio_velocity_sigma=learned_predictions["airio_velocity_inekf"].body_sigma,
        airio_displacement_prediction=learned_predictions["airio_displacement_inekf"].body_velocity,
        airio_displacement_sigma=learned_predictions["airio_displacement_inekf"].body_sigma,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
