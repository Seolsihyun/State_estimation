"""Loader and synchronization helpers for the 11-run CF231 CSV dataset."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

GRAVITY_MAGNITUDE = 9.80665


@dataclass(frozen=True)
class Cf231Run:
    run_id: int
    imu_time: np.ndarray
    imu: np.ndarray
    pose_time: np.ndarray
    position: np.ndarray
    rotation: Rotation
    source_dir: Path


@dataclass(frozen=True)
class SynchronizedRun:
    run_id: int
    time: np.ndarray
    imu: np.ndarray
    position: np.ndarray
    rotation: Rotation
    velocity: np.ndarray
    acceleration: np.ndarray
    angular_velocity_body: np.ndarray
    source_dir: Path


def available_run_ids(root: Path) -> list[int]:
    root = Path(root)
    return sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def load_cf231_run(root: Path, run_id: int) -> Cf231Run:
    source = Path(root) / str(run_id)
    imu_path = source / "cf231_imu_raw.csv"
    pose_path = source / "poses.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(f"Run {run_id} has no IMU CSV: {imu_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Run {run_id} has no pose CSV: {pose_path}")

    imu_ns, imu_values = _load_imu(imu_path)
    pose_ns, position, quaternion_xyzw = _load_pose(pose_path)
    common_origin_ns = min(int(imu_ns[0]), int(pose_ns[0]))
    imu_time = (imu_ns - common_origin_ns).astype(np.float64) * 1.0e-9
    pose_time = (pose_ns - common_origin_ns).astype(np.float64) * 1.0e-9
    return Cf231Run(
        run_id=run_id,
        imu_time=imu_time,
        imu=imu_values,
        pose_time=pose_time,
        position=position,
        rotation=Rotation.from_quat(quaternion_xyzw),
        source_dir=source,
    )


def synchronize_to_imu(run: Cf231Run) -> SynchronizedRun:
    """Interpolate mocap pose and smoothed derivatives at IMU timestamps."""
    start = max(float(run.imu_time[0]), float(run.pose_time[0]))
    end = min(float(run.imu_time[-1]), float(run.pose_time[-1]))
    mask = (run.imu_time >= start) & (run.imu_time <= end)
    query = run.imu_time[mask]
    imu = run.imu[mask]
    if query.size < 10:
        raise ValueError(f"Run {run.run_id} has fewer than 10 synchronized samples.")

    pose_time, unique = np.unique(run.pose_time, return_index=True)
    position = run.position[unique]
    pose_rotation = run.rotation[unique]
    if pose_time.size < 7:
        raise ValueError(f"Run {run.run_id} has too few unique pose samples.")

    smoothed_position, pose_velocity, pose_acceleration = _smooth_kinematics(
        pose_time,
        position,
    )
    position_query = _interp_columns(pose_time, smoothed_position, query)
    velocity_query = _interp_columns(pose_time, pose_velocity, query)
    acceleration_query = _interp_columns(pose_time, pose_acceleration, query)

    rotation_query = Slerp(pose_time, pose_rotation)(query)
    omega_time, omega_body = _body_angular_velocity(pose_time, pose_rotation)
    omega_query = _interp_columns(omega_time, omega_body, query)
    return SynchronizedRun(
        run_id=run.run_id,
        time=query - query[0],
        imu=imu,
        position=position_query,
        rotation=rotation_query,
        velocity=velocity_query,
        acceleration=acceleration_query,
        angular_velocity_body=omega_query,
        source_dir=run.source_dir,
    )


def low_dynamics_mask(
    run: SynchronizedRun,
    *,
    max_speed_m_s: float = 0.08,
    max_acceleration_m_s2: float = 0.8,
    max_angular_rate_rad_s: float = 0.35,
) -> np.ndarray:
    """Select samples whose GT motion is reliable for constant-bias fitting."""
    speed = np.linalg.norm(run.velocity, axis=1)
    acceleration = np.linalg.norm(run.acceleration, axis=1)
    angular_rate = np.linalg.norm(run.angular_velocity_body, axis=1)
    finite = (
        np.all(np.isfinite(run.imu), axis=1)
        & np.all(np.isfinite(run.position), axis=1)
        & np.all(np.isfinite(run.velocity), axis=1)
        & np.all(np.isfinite(run.acceleration), axis=1)
        & np.all(np.isfinite(run.angular_velocity_body), axis=1)
    )
    return (
        finite
        & (speed <= max_speed_m_s)
        & (acceleration <= max_acceleration_m_s2)
        & (angular_rate <= max_angular_rate_rad_s)
    )


def imu_gt_residuals(
    run: SynchronizedRun,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gyro and accelerometer residuals in SI units."""
    rotations = run.rotation.as_matrix()
    gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    specific_force = np.einsum(
        "nji,nj->ni",
        rotations,
        run.acceleration - gravity[None, :],
    )
    gyro_residual = run.imu[:, 3:6] - run.angular_velocity_body
    accel_residual = run.imu[:, 0:3] - specific_force
    if mask is None:
        return gyro_residual, accel_residual
    selected = np.asarray(mask, dtype=bool)
    return gyro_residual[selected], accel_residual[selected]


def _load_imu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamp_ns: list[int] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw = row["values"]
            start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end <= start:
                raise ValueError(f"Cannot parse IMU values in {path}: {raw}")
            sample = ast.literal_eval(raw[start : end + 1])
            if len(sample) != 6:
                raise ValueError(f"Expected 6 IMU values in {path}, got {len(sample)}.")
            timestamp_ns.append(int(row["timestamp_ns"]))
            values.append([float(value) for value in sample])
    time, samples = _strictly_increasing(timestamp_ns, values)
    # Crazyflie logs acceleration in g and angular rate in degrees/second.
    samples[:, 0:3] *= GRAVITY_MAGNITUDE
    samples[:, 3:6] = np.deg2rad(samples[:, 3:6])
    return time, samples


def _load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamp_ns: list[int] = []
    position: list[list[float]] = []
    quaternion: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamp_ns.append(int(row["timestamp_ns"]))
            position.append(
                [
                    float(row["poses.0.pose.position.x"]),
                    float(row["poses.0.pose.position.y"]),
                    float(row["poses.0.pose.position.z"]),
                ]
            )
            quaternion.append(
                [
                    float(row["poses.0.pose.orientation.x"]),
                    float(row["poses.0.pose.orientation.y"]),
                    float(row["poses.0.pose.orientation.z"]),
                    float(row["poses.0.pose.orientation.w"]),
                ]
            )
    time, combined = _strictly_increasing(
        timestamp_ns,
        np.hstack((position, quaternion)),
    )
    return time, combined[:, :3], combined[:, 3:7]


def _strictly_increasing(
    timestamp_ns: list[int],
    values: list[list[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(timestamp_ns, dtype=np.int64)
    samples = np.asarray(values, dtype=float)
    order = np.argsort(time, kind="stable")
    time = time[order]
    samples = samples[order]
    keep = np.r_[True, np.diff(time) > 0]
    return time[keep], samples[keep]


def _smooth_kinematics(
    time: np.ndarray,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median_dt = float(np.median(np.diff(time)))
    nominal_window = max(7, int(round(0.51 / median_dt)))
    if nominal_window % 2 == 0:
        nominal_window += 1
    maximum_window = len(time) if len(time) % 2 == 1 else len(time) - 1
    window = min(nominal_window, maximum_window)
    if window < 7:
        smoothed = position.copy()
    else:
        smoothed = savgol_filter(
            position,
            window_length=window,
            polyorder=min(3, window - 1),
            axis=0,
            mode="interp",
        )
    edge_order = 2 if len(time) >= 3 else 1
    velocity = np.gradient(smoothed, time, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=edge_order)
    return smoothed, velocity, acceleration


def _body_angular_velocity(
    time: np.ndarray,
    rotation: Rotation,
) -> tuple[np.ndarray, np.ndarray]:
    relative = rotation[:-1].inv() * rotation[1:]
    dt = np.diff(time)
    omega = relative.as_rotvec() / dt[:, None]
    midpoint = 0.5 * (time[:-1] + time[1:])
    if omega.shape[0] >= 7:
        median_dt = float(np.median(np.diff(midpoint)))
        nominal_window = max(7, int(round(0.21 / median_dt)))
        if nominal_window % 2 == 0:
            nominal_window += 1
        maximum_window = len(midpoint) if len(midpoint) % 2 == 1 else len(midpoint) - 1
        window = min(nominal_window, maximum_window)
        if window >= 7:
            omega = savgol_filter(
                omega,
                window_length=window,
                polyorder=min(3, window - 1),
                axis=0,
                mode="interp",
            )
    return midpoint, omega


def _interp_columns(
    source_time: np.ndarray,
    values: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_time, source_time, values[:, axis]) for axis in range(values.shape[1])]
    )
