"""Load and synchronize EuRoC IMU and state ground-truth CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class EurocData:
    """EuRoC data represented at IMU timestamps.

    Conventions:
        - ``rotation`` maps the IMU/sensor frame to the reference/world frame.
        - ``position`` and ``velocity`` are expressed in the reference frame.
        - ``imu`` is ordered ``[ax, ay, az, gx, gy, gz]``.
        - timestamps are seconds relative to the first synchronized sample.
    """

    time: np.ndarray
    imu: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    rotation: Rotation
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    imu_metadata: dict[str, Any]
    gt_metadata: dict[str, Any]
    source_dir: Path


def load_euroc_sequence(sequence_dir: str | Path) -> EurocData:
    """Load ``mav0`` or a sequence directory containing ``mav0``."""
    root = Path(sequence_dir).expanduser().resolve()
    mav0 = root / "mav0" if (root / "mav0").is_dir() else root
    imu_csv = mav0 / "imu0" / "data.csv"
    gt_csv = mav0 / "state_groundtruth_estimate0" / "data.csv"
    imu_yaml = mav0 / "imu0" / "sensor.yaml"
    gt_yaml = mav0 / "state_groundtruth_estimate0" / "sensor.yaml"
    required = [imu_csv, gt_csv, imu_yaml, gt_yaml]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing EuRoC files:\n" + "\n".join(missing))

    imu_header, imu_values = _read_named_csv(imu_csv)
    gt_header, gt_values = _read_named_csv(gt_csv)
    imu_columns = _column_map(imu_header)
    gt_columns = _column_map(gt_header)

    imu_ns = _take(imu_values, imu_columns, ["timestamp"])
    gyro = _take_xyz(imu_values, imu_columns, "w_RS_S")
    accel = _take_xyz(imu_values, imu_columns, "a_RS_S")

    gt_ns = _take(gt_values, gt_columns, ["timestamp"])
    position = _take_xyz(gt_values, gt_columns, "p_RS_R")
    velocity = _take_xyz(gt_values, gt_columns, "v_RS_R")
    gyro_bias = _take_xyz(gt_values, gt_columns, "b_w_RS_S")
    accel_bias = _take_xyz(gt_values, gt_columns, "b_a_RS_S")
    quaternion_wxyz = _take(
        gt_values,
        gt_columns,
        ["q_RS_w", "q_RS_x", "q_RS_y", "q_RS_z"],
    )

    imu_seconds = imu_ns * 1e-9
    gt_seconds = gt_ns * 1e-9
    _validate_time(imu_seconds, "IMU")
    _validate_time(gt_seconds, "GT")

    overlap_start = max(imu_seconds[0], gt_seconds[0])
    overlap_end = min(imu_seconds[-1], gt_seconds[-1])
    if overlap_end <= overlap_start:
        raise ValueError("IMU and ground truth have no overlapping timestamps.")
    keep = (imu_seconds >= overlap_start) & (imu_seconds <= overlap_end)
    master_time = imu_seconds[keep]
    gyro, accel = gyro[keep], accel[keep]

    gt_rotation = Rotation.from_quat(quaternion_wxyz[:, [1, 2, 3, 0]])
    synchronized_rotation = Slerp(gt_seconds, gt_rotation)(master_time)
    synchronized_position = _interp_vectors(gt_seconds, position, master_time)
    synchronized_velocity = _interp_vectors(gt_seconds, velocity, master_time)
    synchronized_bg = _interp_vectors(gt_seconds, gyro_bias, master_time)
    synchronized_ba = _interp_vectors(gt_seconds, accel_bias, master_time)

    time = master_time - master_time[0]
    imu = np.column_stack([accel, gyro])
    if not all(
        np.all(np.isfinite(values))
        for values in [
            time,
            imu,
            synchronized_position,
            synchronized_velocity,
            synchronized_rotation.as_quat(),
            synchronized_bg,
            synchronized_ba,
        ]
    ):
        raise ValueError("Synchronized EuRoC data contain non-finite values.")

    return EurocData(
        time=time,
        imu=imu,
        position=synchronized_position,
        velocity=synchronized_velocity,
        rotation=synchronized_rotation,
        gyro_bias=synchronized_bg,
        accel_bias=synchronized_ba,
        imu_metadata=_read_yaml(imu_yaml),
        gt_metadata=_read_yaml(gt_yaml),
        source_dir=mav0,
    )


def _read_named_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig") as stream:
        header_line = stream.readline().strip()
    if not header_line.startswith("#"):
        raise ValueError(f"Expected a commented EuRoC CSV header in {path}.")
    header = [item.strip().lstrip("#").strip() for item in header_line.split(",")]
    values = np.loadtxt(path, delimiter=",", comments="#", dtype=float)
    values = np.atleast_2d(values)
    if values.shape[1] != len(header):
        raise ValueError(
            f"{path}: header has {len(header)} columns but data have {values.shape[1]}."
        )
    return header, values


def _normalize_column(name: str) -> str:
    # EuRoC headers append units such as "[ns]" or "[rad s^-1]".
    return name.split("[", 1)[0].strip().replace(" ", "")


def _column_map(header: list[str]) -> dict[str, int]:
    result = {_normalize_column(name): index for index, name in enumerate(header)}
    if len(result) != len(header):
        raise ValueError(f"Duplicate normalized CSV columns: {header}")
    return result


def _resolve_column(columns: dict[str, int], requested: str) -> int:
    normalized = _normalize_column(requested)
    if normalized in columns:
        return columns[normalized]
    # The first EuRoC column is commonly "#timestamp" with a unit suffix.
    if normalized == "timestamp":
        candidates = [index for name, index in columns.items() if "timestamp" in name.lower()]
        if len(candidates) == 1:
            return candidates[0]
    raise KeyError(f"Missing EuRoC CSV column '{requested}'. Available: {list(columns)}")


def _take(values: np.ndarray, columns: dict[str, int], names: list[str]) -> np.ndarray:
    indices = [_resolve_column(columns, name) for name in names]
    selected = values[:, indices]
    return selected[:, 0] if len(indices) == 1 else selected


def _take_xyz(values: np.ndarray, columns: dict[str, int], prefix: str) -> np.ndarray:
    return _take(values, columns, [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])


def _interp_vectors(source_time, source_values, target_time) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_time, source_time, source_values[:, axis])
            for axis in range(source_values.shape[1])
        ]
    )


def _validate_time(time: np.ndarray, label: str) -> None:
    if time.ndim != 1 or len(time) < 2:
        raise ValueError(f"{label} timestamps must be a vector with at least two samples.")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{label} timestamps must be finite and strictly increasing.")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value if isinstance(value, dict) else {}
