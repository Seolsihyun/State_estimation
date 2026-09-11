from __future__ import annotations

import numpy as np

from models.Hoon_lie_group_utils import (
    as_matrix,
    exp_se23,
    exp_so3,
    gamma2_so3,
    hat_so3,
    left_jacobian_so3,
)
from utils.math_utils import wrap_angle
from utils.rotation_utils import rot_to_rpy, rpy_to_rot


# Backward-compatible names used by the filters in this repository.
skew = hat_so3
so3_exp = exp_so3
so3_left_jacobian = left_jacobian_so3
se23_exp = exp_se23


def from_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from models.Hoon_lie_group_utils import from_matrix as _from_matrix

    return _from_matrix(X)


def pose_to_state(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(mean, dtype=float).reshape(6)
    return pose[0:3].copy(), rpy_to_rot(pose[3:6])


def pose_from_state(Rot: np.ndarray, p: np.ndarray) -> np.ndarray:
    rpy = np.array([wrap_angle(angle) for angle in rot_to_rpy(Rot)], dtype=float)
    return np.concatenate([np.asarray(p, dtype=float).reshape(3), rpy])


def propagate_mean(
    Rot: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    control: np.ndarray,
    dt: float,
    *,
    use_imu_velocity: bool,
    use_imu_rotation: bool,
    translation_input_frame: str,
    translation_input_type: str,
    rotation_input_type: str,
    rotation_representation: str,
    gravity: np.ndarray,
    velocity_blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rot = np.asarray(Rot, dtype=float).reshape(3, 3)
    v = np.asarray(v, dtype=float).reshape(3).copy()
    p = np.asarray(p, dtype=float).reshape(3).copy()
    control = np.asarray(control, dtype=float).reshape(-1)
    dt = float(dt)

    R_prev = Rot
    if use_imu_rotation:
        rot_vec = control[3:6]
        if rotation_input_type == "rate":
            rot_vec = rot_vec * dt
        elif rotation_input_type != "increment":
            raise ValueError(f"Unsupported rotation_input_type: {rotation_input_type}")

        if rotation_representation == "rotvec":
            Rot = R_prev @ exp_so3(rot_vec)
        elif rotation_representation == "euler":
            Rot = rpy_to_rot(rot_to_rpy(R_prev) + rot_vec)
        else:
            raise ValueError(f"Unsupported rotation_representation: {rotation_representation}")

    if not use_imu_velocity:
        return Rot, v, p + v * dt

    trans = control[0:3]
    if translation_input_frame == "body":
        if translation_input_type == "acceleration":
            rot_increment = control[3:6] * dt if rotation_input_type == "rate" else control[3:6]
            trans_world = R_prev @ left_jacobian_so3(rot_increment) @ trans
        else:
            trans_world = R_prev @ trans
    elif translation_input_frame == "world":
        trans_world = trans
    else:
        raise ValueError(f"Unsupported translation_input_frame: {translation_input_frame}")

    if translation_input_type == "velocity":
        v = velocity_blend * v + (1.0 - velocity_blend) * trans_world
        p = p + v * dt
    elif translation_input_type == "acceleration":
        accel_world = trans_world + np.asarray(gravity, dtype=float).reshape(3)
        if translation_input_frame == "body":
            rot_increment = control[3:6] * dt if rotation_input_type == "rate" else control[3:6]
            p = p + v * dt + R_prev @ gamma2_so3(rot_increment) @ trans * dt**2 + 0.5 * np.asarray(gravity, dtype=float).reshape(3) * dt**2
        else:
            p = p + v * dt + 0.5 * accel_world * dt**2
        v = v + accel_world * dt
    elif translation_input_type == "increment":
        p = p + trans_world
        if dt > 1e-12:
            inferred_v = trans_world / dt
            v = velocity_blend * v + (1.0 - velocity_blend) * inferred_v
    else:
        raise ValueError(f"Unsupported translation_input_type: {translation_input_type}")
    return Rot, v, p


def error_transition(dt: float) -> np.ndarray:
    Phi = np.eye(9, dtype=float)
    Phi[6:9, 3:6] = np.eye(3, dtype=float) * float(dt)
    return Phi


def position_measurement_matrix(p: np.ndarray, indices: np.ndarray) -> np.ndarray:
    H = np.zeros((len(indices), 9), dtype=float)
    p = np.asarray(p, dtype=float).reshape(3)
    for row, idx in enumerate(np.asarray(indices, dtype=int)):
        H[row, 0:3] = -hat_so3(p)[idx, :]
        H[row, 6 + idx] = 1.0
    return H
