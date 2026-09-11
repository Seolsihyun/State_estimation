from __future__ import annotations

from typing import Iterable

import numpy as np

from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import exp_so3, gamma2_so3, left_jacobian_so3, log_so3, symmetrize_covariance
from utils.filter_math import diagonal_covariance
from utils.math_utils import fit_diag, fit_vector
from utils.sigma_points import merwe_sigma_points


class QuaternionUnscentedKalmanFilter15D:
    """Quaternion UKF for nominal state [p, v, q, bg, ba].

    The quaternion is a unit-norm nominal attitude representation, while the
    covariance remains the 15D tangent error state [dtheta, dv, dp, dbg, dba].
    This avoids Euler-angle averaging without storing a rotation matrix.
    """

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        initialization_config: dict | None = None,
        sigma_point_config: dict | None = None,
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type != "3d":
            raise ValueError("QuaternionUnscentedKalmanFilter15D supports only 3d pose.")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        init_cfg = initialization_config or {}
        sigma_cfg = sigma_point_config or {}

        self.pose_type = pose_type
        self.mode = mode
        self.error_dim = 15
        self.translation_input_frame = str(motion_cfg.get("translation_input_frame", "body"))
        self.translation_input_type = str(motion_cfg.get("translation_input_type", "acceleration"))
        self.rotation_input_type = str(motion_cfg.get("rotation_input_type", "rate"))
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.process_noise_diag = fit_diag(
            motion_cfg.get(
                "process_noise_diag",
                [1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5],
            ),
            self.error_dim,
        )
        self.measurement_noise_diag = fit_diag(meas_cfg.get("measurement_noise_diag", [1.0, 1.0, 1.0]), 3)
        self.velocity_measurement_noise_diag = fit_diag(
            meas_cfg.get("velocity_measurement_noise_diag", self.measurement_noise_diag),
            3,
        )
        self.alpha = float(sigma_cfg.get("alpha", 0.35))
        self.beta = float(sigma_cfg.get("beta", 2.0))
        self.kappa = float(sigma_cfg.get("kappa", 0.0))
        self.covariance_floor = float(motion_cfg.get("covariance_floor", 1.0e-12))
        self.covariance_ceiling = float(motion_cfg.get("covariance_ceiling", 1.0e8))
        self.max_delta_norm = float(motion_cfg.get("max_delta_norm", 100.0))

        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.gyro_bias = fit_vector(motion_cfg.get("gyro_bias", [0.0, 0.0, 0.0]), 3)
        self.accel_bias = fit_vector(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), 3)
        self.P = np.eye(self.error_dim, dtype=float)
        self.Q = diagonal_covariance(self.process_noise_diag)
        self.innovation = np.zeros(3, dtype=float)
        self.K = np.zeros((self.error_dim, 3), dtype=float)
        self.initialized = False
        self.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"), init_cfg.get("velocity_mean"))

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "QuaternionUnscentedKalmanFilter15D":
        cfg = compare_config.get(
            "quaternion_ukf_15d",
            compare_config.get(
                "Quaternion_ukf_15d",
                compare_config.get("manifold_ukf_15d", compare_config.get("Manifold_ukf_15d", compare_config)),
            ),
        )
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
            sigma_point_config=cfg.get("sigma_points", {}),
        )

    def initialize(
        self,
        mean: Iterable[float] | None = None,
        cov_diag: Iterable[float] | None = None,
        velocity_mean: Iterable[float] | None = None,
    ) -> None:
        pose = fit_vector(np.zeros(6) if mean is None else np.asarray(mean, dtype=float).reshape(-1), 6)
        self.p, Rot0 = lie.pose_to_state(pose)
        self.q = _quat_from_rot(Rot0)
        self.v = fit_vector(np.zeros(3) if velocity_mean is None else np.asarray(velocity_mean, dtype=float).reshape(-1), 3)
        cov = fit_diag(np.ones(self.error_dim) * 1e-3 if cov_diag is None else cov_diag, self.error_dim)
        self.P = diagonal_covariance(cov)
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        if control is None:
            return self.estimate_pose()

        sigmas, Wm, Wc, _ = merwe_sigma_points(np.zeros(self.error_dim), self.P, self.alpha, self.beta, self.kappa)
        propagated = [self._propagate_nominal(*self._compose(delta), control, dt) for delta in sigmas]
        self.q = _quaternion_mean([item[0] for item in propagated], Wm, self.q)
        self.v = np.sum([Wm[i] * propagated[i][1] for i in range(len(propagated))], axis=0)
        self.p = np.sum([Wm[i] * propagated[i][2] for i in range(len(propagated))], axis=0)
        self.gyro_bias = np.sum([Wm[i] * propagated[i][3] for i in range(len(propagated))], axis=0)
        self.accel_bias = np.sum([Wm[i] * propagated[i][4] for i in range(len(propagated))], axis=0)

        errors = np.vstack([self._state_error(*item) for item in propagated])
        self.Q = diagonal_covariance(self.process_noise_diag)
        P = _weighted_outer(errors, Wc) + self.Q * max(float(dt), 1e-9)
        self.P = self._stabilize(P)
        return self.estimate_pose()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self._update_linear_measurement(z, kind="position")
        return self.estimate_pose()

    def velocity_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.estimate_pose()
        z = np.asarray(measurement, dtype=float).reshape(3)
        self._update_linear_measurement(z, kind="velocity")
        return self.estimate_pose()

    def step(
        self,
        control: Iterable[float] | None,
        measurement: Iterable[float] | None,
        dt: float,
        mode: str | None = None,
    ) -> np.ndarray:
        run_mode = self.mode if mode is None else mode
        if run_mode in {"imu_only", "fused"}:
            self.predict(control, dt)
        if run_mode in {"gnss_only", "fused"}:
            self.measurement_update(measurement)
        return self.estimate_pose()

    def estimate_pose(self) -> np.ndarray:
        return lie.pose_from_state(_rot_from_quat(self.q), self.p)

    def _compose(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta = np.asarray(delta, dtype=float).reshape(self.error_dim)
        q = _quat_multiply(self.q, _quat_from_rotvec(delta[0:3]))
        v = self.v + delta[3:6]
        p = self.p + delta[6:9]
        bg = self.gyro_bias + delta[9:12]
        ba = self.accel_bias + delta[12:15]
        return q, v, p, bg, ba

    def _state_error(self, q: np.ndarray, v: np.ndarray, p: np.ndarray, bg: np.ndarray, ba: np.ndarray) -> np.ndarray:
        R_ref = _rot_from_quat(self.q)
        R = _rot_from_quat(q)
        return np.concatenate([
            log_so3(R_ref.T @ R),
            np.asarray(v) - self.v,
            np.asarray(p) - self.p,
            np.asarray(bg) - self.gyro_bias,
            np.asarray(ba) - self.accel_bias,
        ])

    def _propagate_nominal(
        self,
        q: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        control: Iterable[float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = np.asarray(control, dtype=float).reshape(-1)
        if u.size < 6:
            raise ValueError("3D Quaternion UKF control must contain [ax, ay, az, gx, gy, gz].")
        dt = max(float(dt), 0.0)
        R_prev = _rot_from_quat(q)
        v_prev = np.asarray(v, dtype=float).reshape(3)
        p_prev = np.asarray(p, dtype=float).reshape(3)
        w = u[3:6] - np.asarray(gyro_bias, dtype=float).reshape(3)
        a = u[0:3] - np.asarray(accel_bias, dtype=float).reshape(3)
        phi = w * dt if self.rotation_input_type == "rate" else w
        q_next = _quat_multiply(q, _quat_from_rotvec(phi))
        if self.translation_input_type != "acceleration":
            raise ValueError("Quaternion UKF currently supports acceleration input only.")
        if self.translation_input_frame == "body":
            accel_world = R_prev @ left_jacobian_so3(phi) @ a + self.gravity
            p_next = p_prev + v_prev * dt + R_prev @ gamma2_so3(phi) @ a * dt * dt + 0.5 * self.gravity * dt * dt
        elif self.translation_input_frame == "world":
            accel_world = a + self.gravity
            p_next = p_prev + v_prev * dt + 0.5 * accel_world * dt * dt
        else:
            raise ValueError(f"Unsupported translation_input_frame: {self.translation_input_frame}")
        v_next = v_prev + accel_world * dt
        return q_next, v_next, p_next, np.asarray(gyro_bias).copy(), np.asarray(accel_bias).copy()

    def _update_linear_measurement(self, z: np.ndarray, kind: str) -> None:
        sigmas, Wm, Wc, _ = merwe_sigma_points(np.zeros(self.error_dim), self.P, self.alpha, self.beta, self.kappa)
        states = [self._compose(delta) for delta in sigmas]
        if kind == "position":
            values = np.vstack([state[2] for state in states])
            z_mean = np.sum(values * Wm[:, None], axis=0)
            Rm = diagonal_covariance(self.measurement_noise_diag)
        elif kind == "velocity":
            values = np.vstack([state[1] for state in states])
            z_mean = np.sum(values * Wm[:, None], axis=0)
            Rm = diagonal_covariance(self.velocity_measurement_noise_diag)
        else:
            raise ValueError(f"Unsupported measurement kind: {kind}")

        errors = np.vstack([self._state_error(*state) for state in states])
        dz = values - z_mean
        S = _weighted_outer(dz, Wc) + Rm + 1e-12 * np.eye(3)
        Pxz = np.zeros((self.error_dim, 3), dtype=float)
        for i in range(len(sigmas)):
            Pxz += Wc[i] * np.outer(errors[i], dz[i])
        self.K = Pxz @ np.linalg.inv(S)
        self.innovation = z - z_mean
        delta = self._bounded_delta(self.K @ self.innovation)
        self.q = _quat_multiply(self.q, _quat_from_rotvec(delta[0:3]))
        self.v = self.v + delta[3:6]
        self.p = self.p + delta[6:9]
        self.gyro_bias = self.gyro_bias + delta[9:12]
        self.accel_bias = self.accel_bias + delta[12:15]
        self.P = self._stabilize(self.P - self.K @ S @ self.K.T)

    def _bounded_delta(self, delta: np.ndarray) -> np.ndarray:
        delta = np.nan_to_num(np.asarray(delta, dtype=float).reshape(self.error_dim), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(delta))
        if self.max_delta_norm > 0.0 and norm > self.max_delta_norm:
            delta = delta * (self.max_delta_norm / norm)
        return delta

    def _stabilize(self, P: np.ndarray) -> np.ndarray:
        return symmetrize_covariance(P, floor=self.covariance_floor, ceiling=self.covariance_ceiling)


def _weighted_outer(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cov = np.zeros((values.shape[1], values.shape[1]), dtype=float)
    for i in range(values.shape[0]):
        cov += weights[i] * np.outer(values[i], values[i])
    return 0.5 * (cov + cov.T)


def _quaternion_mean(quaternions: list[np.ndarray], weights: np.ndarray, initial: np.ndarray) -> np.ndarray:
    mean = _quat_normalize(initial)
    for _ in range(12):
        R_mean = _rot_from_quat(mean)
        delta = np.zeros(3, dtype=float)
        for q, w in zip(quaternions, weights):
            delta += w * log_so3(R_mean.T @ _rot_from_quat(q))
        if np.linalg.norm(delta) < 1.0e-10:
            break
        mean = _quat_multiply(mean, _quat_from_rotvec(delta))
    return _quat_normalize(mean)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / norm
    return -q if q[0] < 0.0 else q


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _quat_normalize(q1)
    w2, x2, y2, z2 = _quat_normalize(q2)
    return _quat_normalize(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float))


def _quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    if angle < 1.0e-12:
        return _quat_normalize(np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]], dtype=float))
    axis = phi / angle
    half = 0.5 * angle
    return _quat_normalize(np.concatenate([[np.cos(half)], axis * np.sin(half)]))


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def _quat_from_rot(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return _quat_normalize(q)
