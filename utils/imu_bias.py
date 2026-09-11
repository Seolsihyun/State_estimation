"""Ground-truth-based IMU bias calibration for dead reckoning.

Rotations use the body-to-world convention used by the filters in this
repository.  Accelerometer measurements are body-frame specific force and
gyroscope measurements are body-frame angular rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from models.Hoon_lie_group_utils import log_so3


@dataclass(frozen=True)
class ImuBiasEstimate:
    gyro: np.ndarray
    accel: np.ndarray
    sample_count: int
    gyro_residual_std: np.ndarray
    accel_residual_std: np.ndarray


def estimate_imu_bias_from_gt(
    timestamps: Iterable[float],
    imu: np.ndarray,
    gt_rotations: np.ndarray,
    gt_velocities: np.ndarray | None = None,
    *,
    gt_positions: np.ndarray | None = None,
    gravity: Iterable[float] = (0.0, 0.0, -9.81),
    calibration_mask: np.ndarray | None = None,
    calibration_range: tuple[float, float] | None = None,
    statistic: Literal["mean", "median"] = "mean",
) -> ImuBiasEstimate:
    """Estimate constant gyro/accelerometer bias from synchronized GT.

    Args:
        timestamps: Strictly increasing timestamps in seconds, shape ``(N,)``.
        imu: ``[ax, ay, az, gx, gy, gz]`` at each timestamp, shape ``(N, 6)``.
        gt_rotations: Body-to-world rotation matrices, shape ``(N, 3, 3)``.
        gt_velocities: World-frame GT velocity, shape ``(N, 3)``. If it is not
            available, pass ``None`` and provide ``gt_positions``.
        gt_positions: Optional world-frame GT positions, shape ``(N, 3)``.
            Velocity is numerically differentiated from these positions only
            when ``gt_velocities`` is ``None``.
        gravity: World-frame gravity vector, identical to the filter setting.
        calibration_mask: Optional boolean sample mask, shape ``(N,)``.
        calibration_range: Optional inclusive time range ``(start, end)``.
        statistic: Mean for the constant-bias least-squares solution, or median
            for a robust estimate in the presence of outliers.

    The residuals are formed on intervals.  IMU samples and GT rotations are
    represented at interval midpoints; GT acceleration is the velocity finite
    difference over the same interval.
    """
    t = np.asarray(timestamps, dtype=float).reshape(-1)
    u = np.asarray(imu, dtype=float)
    rotations = np.asarray(gt_rotations, dtype=float)
    n = t.size
    if n < 2:
        raise ValueError("At least two synchronized IMU/GT samples are required.")
    if not np.all(np.isfinite(t)):
        raise ValueError("timestamps must contain only finite values.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("timestamps must be strictly increasing.")

    if gt_velocities is None:
        if gt_positions is None:
            raise ValueError("Provide either gt_velocities or gt_positions.")
        positions = np.asarray(gt_positions, dtype=float)
        if positions.shape != (n, 3):
            raise ValueError(f"gt_positions must have shape ({n}, 3), got {positions.shape}.")
        velocities = np.gradient(positions, t, axis=0, edge_order=2 if n >= 3 else 1)
    else:
        velocities = np.asarray(gt_velocities, dtype=float)
    g = np.asarray(gravity, dtype=float).reshape(3)

    if u.shape != (n, 6):
        raise ValueError(f"imu must have shape ({n}, 6), got {u.shape}.")
    if rotations.shape != (n, 3, 3):
        raise ValueError(f"gt_rotations must have shape ({n}, 3, 3), got {rotations.shape}.")
    if velocities.shape != (n, 3):
        raise ValueError(f"gt_velocities must have shape ({n}, 3), got {velocities.shape}.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(u)) and np.all(np.isfinite(rotations)) and np.all(np.isfinite(velocities))):
        raise ValueError("Calibration inputs must contain only finite values.")

    sample_mask = np.ones(n, dtype=bool)
    if calibration_mask is not None:
        supplied_mask = np.asarray(calibration_mask, dtype=bool).reshape(-1)
        if supplied_mask.shape != (n,):
            raise ValueError(f"calibration_mask must have shape ({n},).")
        sample_mask &= supplied_mask
    if calibration_range is not None:
        start, end = map(float, calibration_range)
        if end < start:
            raise ValueError("calibration_range end must be greater than or equal to start.")
        sample_mask &= (t >= start) & (t <= end)

    # Both ends must belong to the calibration window so no interval crosses
    # its boundary.
    interval_mask = sample_mask[:-1] & sample_mask[1:]
    if not np.any(interval_mask):
        raise ValueError("The selected calibration window contains no complete interval.")

    imu_mid = 0.5 * (u[:-1] + u[1:])
    accel_world = np.diff(velocities, axis=0) / dt[:, None]
    rotation_mid = np.empty((n - 1, 3, 3), dtype=float)
    omega_gt = np.empty((n - 1, 3), dtype=float)
    for i in range(n - 1):
        relative = rotations[i].T @ rotations[i + 1]
        rotvec = log_so3(relative)
        omega_gt[i] = rotvec / dt[i]
        rotation_mid[i] = rotations[i] @ _so3_exp(0.5 * rotvec)

    specific_force_gt = np.einsum(
        "nji,nj->ni", rotation_mid, accel_world - g[None, :]
    )
    gyro_residuals = (imu_mid[:, 3:6] - omega_gt)[interval_mask]
    accel_residuals = (imu_mid[:, 0:3] - specific_force_gt)[interval_mask]

    reducer = np.mean if statistic == "mean" else np.median if statistic == "median" else None
    if reducer is None:
        raise ValueError("statistic must be 'mean' or 'median'.")
    return ImuBiasEstimate(
        gyro=reducer(gyro_residuals, axis=0),
        accel=reducer(accel_residuals, axis=0),
        sample_count=int(np.count_nonzero(interval_mask)),
        gyro_residual_std=np.std(gyro_residuals, axis=0),
        accel_residual_std=np.std(accel_residuals, axis=0),
    )


def apply_fixed_imu_bias(estimator: object, estimate: ImuBiasEstimate) -> None:
    """Apply a calibrated bias and disable its random walk on a 15D filter.

    This supports the EKF, InEKF, UKF, and PF implementations in this repo.
    Run the estimator with ``mode='imu_only'`` after calling this function.
    """
    gyro = np.asarray(estimate.gyro, dtype=float).reshape(3)
    accel = np.asarray(estimate.accel, dtype=float).reshape(3)

    if hasattr(estimator, "set_fixed_imu_bias"):
        estimator.set_fixed_imu_bias(gyro, accel)
        return

    if hasattr(estimator, "gyro_bias") and hasattr(estimator, "accel_bias"):
        estimator.gyro_bias = gyro.copy()
        estimator.accel_bias = accel.copy()
    elif hasattr(estimator, "particles_bg") and hasattr(estimator, "particles_ba"):
        estimator.particles_bg[:] = gyro
        estimator.particles_ba[:] = accel
    else:
        raise TypeError("Estimator does not expose a supported 15D IMU-bias state.")

    # Known bias is deterministic during dead reckoning.  Removing covariance
    # and process noise in these blocks prevents UKF/PF bias spread and drift.
    if hasattr(estimator, "process_noise_diag"):
        estimator.process_noise_diag[9:15] = 0.0
    if hasattr(estimator, "P"):
        estimator.P[9:15, :] = 0.0
        estimator.P[:, 9:15] = 0.0
    if hasattr(estimator, "update_biases"):
        estimator.update_biases = False


def _so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = np.array(
        [[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]],
        dtype=float,
    )
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / theta**2) * (K @ K)
