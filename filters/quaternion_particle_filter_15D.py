from __future__ import annotations

from typing import Iterable

import numpy as np

from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import gamma2_so3, left_jacobian_so3
from utils.filter_math import diagonal_gaussian_logpdf
from utils.math_utils import fit_diag, fit_vector
from utils.resampling import resample_indices


class QuaternionParticleFilter15D:
    """Quaternion inertial PF with vectorized batch prediction with particles [p, v, q, bg, ba].

    This keeps the same state and measurement model as QuaternionParticleFilter15D,
    but batches the IMU prediction over all particles to remove the per-particle
    Python loop.
    """

    def __init__(
        self,
        pose_type: str = "3d",
        mode: str = "fused",
        num_particles: int = 4000,
        resample_threshold_ratio: float = 0.5,
        seed: int | None = None,
        motion_config: dict | None = None,
        measurement_config: dict | None = None,
        initialization_config: dict | None = None,
        resampling_method: str = "systematic",
    ) -> None:
        if pose_type == "6d":
            pose_type = "3d"
        if pose_type != "3d":
            raise ValueError("QuaternionParticleFilter15D supports only 3d pose.")

        motion_cfg = motion_config or {}
        meas_cfg = measurement_config or {}
        init_cfg = initialization_config or {}

        self.pose_type = pose_type
        self.mode = mode
        self.dim = 15
        self.num_particles = int(num_particles)
        self.threshold = max(1, int(self.num_particles * float(resample_threshold_ratio)))
        self.rng = np.random.default_rng(seed)
        self.translation_input_frame = str(motion_cfg.get("translation_input_frame", "body"))
        self.translation_input_type = str(motion_cfg.get("translation_input_type", "acceleration"))
        self.rotation_input_type = str(motion_cfg.get("rotation_input_type", "rate"))
        self.gravity = np.asarray(motion_cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.process_noise_diag = fit_diag(
            motion_cfg.get(
                "process_noise_diag",
                [1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5],
            ),
            self.dim,
            fill_missing="zero",
        )
        self.measurement_noise_diag = fit_diag(meas_cfg.get("measurement_noise_diag", [1.0, 1.0, 1.0]), 3)
        self.velocity_measurement_noise_diag = fit_diag(
            meas_cfg.get("velocity_measurement_noise_diag", [0.2, 0.2, 0.2]),
            3,
            fill_missing="edge",
        )
        self.measurement_rejuvenation = bool(meas_cfg.get("measurement_rejuvenation", True))
        self.measurement_rejuvenation_std = fit_diag(
            meas_cfg.get("measurement_rejuvenation_std", np.sqrt(np.clip(self.measurement_noise_diag, 1e-12, None))),
            3,
            fill_missing="edge",
        )
        self.velocity_rejuvenation_std = fit_diag(
            meas_cfg.get("velocity_rejuvenation_std", np.sqrt(np.clip(self.velocity_measurement_noise_diag, 1e-12, None))),
            3,
            fill_missing="edge",
        )
        self.use_pseudo_velocity = bool(meas_cfg.get("use_pseudo_velocity", True))
        self.estimate_method = str(meas_cfg.get("estimate_method", "map")).lower()
        self.resampling_method = str(resampling_method).lower()

        self.particles_p = np.zeros((self.num_particles, 3), dtype=float)
        self.particles_v = np.zeros((self.num_particles, 3), dtype=float)
        self.particles_q = np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float), self.num_particles, axis=0)
        self.particles_bg = np.zeros((self.num_particles, 3), dtype=float)
        self.particles_ba = np.zeros((self.num_particles, 3), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=float)
        self.log_likelihood = np.zeros(self.num_particles, dtype=float)
        self._time_since_measurement = 0.0
        self._last_position_measurement: np.ndarray | None = None
        self.initialized = False
        self.initialize(init_cfg.get("mean"), init_cfg.get("cov_diag"), init_cfg.get("velocity_mean"), motion_cfg)

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "QuaternionParticleFilter15D":
        cfg = compare_config.get(
            "quaternion_pf_15d",
            compare_config.get(
                "Quaternion_pf_15d",
                compare_config.get("batch_quaternion_pf_15d", compare_config.get("Batch_quaternion_pf_15d", compare_config)),
            ),
        )
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            num_particles=cfg.get("num_particles", 4000),
            resample_threshold_ratio=cfg.get("resample_threshold_ratio", 0.5),
            seed=cfg.get("seed"),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
            resampling_method=cfg.get("resampling_method", "systematic"),
        )

    def initialize(
        self,
        mean: Iterable[float] | None = None,
        cov_diag: Iterable[float] | None = None,
        velocity_mean: Iterable[float] | None = None,
        motion_config: dict | None = None,
    ) -> None:
        pose = fit_vector(np.zeros(6) if mean is None else np.asarray(mean, dtype=float).reshape(-1), 6)
        p0, R0 = lie.pose_to_state(pose)
        v0 = fit_vector(np.zeros(3) if velocity_mean is None else np.asarray(velocity_mean, dtype=float).reshape(-1), 3)
        motion_cfg = motion_config or {}
        bg0 = fit_vector(motion_cfg.get("gyro_bias", [0.0, 0.0, 0.0]), 3)
        ba0 = fit_vector(motion_cfg.get("accel_bias", [0.0, 0.0, 0.0]), 3)
        cov = fit_diag(np.zeros(self.dim) if cov_diag is None else cov_diag, self.dim, fill_missing="zero")
        std = np.sqrt(np.clip(cov, 0.0, None))
        tangent = self.rng.normal(0.0, std, size=(self.num_particles, self.dim))
        q0 = _quat_from_rot(R0)
        self.particles_q = np.stack([_quat_multiply(q0, _quat_from_rotvec(tangent[i, 0:3])) for i in range(self.num_particles)], axis=0)
        self.particles_v = v0[None, :] + tangent[:, 3:6]
        self.particles_p = p0[None, :] + tangent[:, 6:9]
        self.particles_bg = bg0[None, :] + tangent[:, 9:12]
        self.particles_ba = ba0[None, :] + tangent[:, 12:15]
        self.weights.fill(1.0 / self.num_particles)
        self._time_since_measurement = 0.0
        self._last_position_measurement = None
        self.initialized = True

    def predict(self, control: Iterable[float] | None, dt: float) -> np.ndarray:
        if not self.initialized:
            self.initialize()
        if control is None:
            return self._state_matrix()
        u = np.asarray(control, dtype=float).reshape(-1)
        if u.size < 6:
            raise ValueError("3D Quaternion PF control must contain [ax, ay, az, gx, gy, gz].")
        dt = max(float(dt), 0.0)
        std = np.sqrt(np.clip(self.process_noise_diag, 0.0, None)) * np.sqrt(max(dt, 1e-12))
        noise = self.rng.normal(0.0, std, size=(self.num_particles, self.dim))

        R_prev = _rot_from_quat_batch(self.particles_q)
        v_prev = self.particles_v.copy()
        p_prev = self.particles_p.copy()
        w = u[3:6][None, :] - self.particles_bg
        a = u[0:3][None, :] - self.particles_ba
        phi = w * dt if self.rotation_input_type == "rate" else w
        if self.translation_input_type != "acceleration":
            raise ValueError("Quaternion PF currently supports acceleration input only.")

        if self.translation_input_frame == "body":
            J = _left_jacobian_so3_batch(phi)
            G2 = _gamma2_so3_batch(phi)
            Ja = np.einsum("nij,nj->ni", J, a)
            G2a = np.einsum("nij,nj->ni", G2, a)
            accel_world = np.einsum("nij,nj->ni", R_prev, Ja) + self.gravity[None, :]
            self.particles_p = p_prev + v_prev * dt + np.einsum("nij,nj->ni", R_prev, G2a) * dt * dt + 0.5 * self.gravity[None, :] * dt * dt
        elif self.translation_input_frame == "world":
            accel_world = a + self.gravity[None, :]
            self.particles_p = p_prev + v_prev * dt + 0.5 * accel_world * dt * dt
        else:
            raise ValueError(f"Unsupported translation_input_frame: {self.translation_input_frame}")

        self.particles_v = v_prev + accel_world * dt
        self.particles_q = _quat_multiply_batch(self.particles_q, _quat_from_rotvec_batch(phi))
        self.particles_q = _quat_multiply_batch(self.particles_q, _quat_from_rotvec_batch(noise[:, 0:3]))
        self.particles_v += noise[:, 3:6]
        self.particles_p += noise[:, 6:9]
        self.particles_bg += noise[:, 9:12]
        self.particles_ba += noise[:, 12:15]
        self._time_since_measurement += dt
        return self._state_matrix()

    def measurement_update(self, measurement: Iterable[float] | None) -> np.ndarray:
        if measurement is None:
            return self.weights
        z = np.asarray(measurement, dtype=float).reshape(3)
        innovation = self.particles_p - z[None, :]
        log_likelihood = diagonal_gaussian_logpdf(innovation, np.clip(self.measurement_noise_diag, 1e-12, None))

        pseudo_v = None
        if self.use_pseudo_velocity and self._last_position_measurement is not None and self._time_since_measurement > 1e-9:
            pseudo_v = (z - self._last_position_measurement) / self._time_since_measurement
            vel_innovation = self.particles_v - pseudo_v[None, :]
            log_likelihood += diagonal_gaussian_logpdf(vel_innovation, np.clip(self.velocity_measurement_noise_diag, 1e-12, None))

        self.log_likelihood = log_likelihood
        self.weights *= np.exp(log_likelihood - np.max(log_likelihood))
        self.normalize()
        if self.effective_sample_size() < self.threshold:
            self.resample()

        if self.measurement_rejuvenation:
            self.particles_p = z[None, :] + self.rng.normal(
                0.0,
                np.asarray(self.measurement_rejuvenation_std, dtype=float).reshape(1, 3),
                size=(self.num_particles, 3),
            )
            if pseudo_v is not None:
                self.particles_v = pseudo_v[None, :] + self.rng.normal(
                    0.0,
                    np.asarray(self.velocity_rejuvenation_std, dtype=float).reshape(1, 3),
                    size=(self.num_particles, 3),
                )
            self.weights.fill(1.0 / self.num_particles)

        self._last_position_measurement = z.copy()
        self._time_since_measurement = 0.0
        return self.weights

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
        if self.effective_sample_size() < self.threshold:
            self.resample()
        return self.estimate_pose()

    def normalize(self) -> np.ndarray:
        total = float(np.sum(self.weights))
        self.weights[:] = 1.0 / self.num_particles if not np.isfinite(total) or total <= 0.0 else self.weights / total
        return self.weights

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def resample(self) -> None:
        indices = resample_indices(self.resampling_method, self.weights, self.rng)
        self.particles_p = self.particles_p[indices]
        self.particles_v = self.particles_v[indices]
        self.particles_q = self.particles_q[indices]
        self.particles_bg = self.particles_bg[indices]
        self.particles_ba = self.particles_ba[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_pose(self) -> np.ndarray:
        if self.estimate_method == "map":
            idx = int(np.argmax(self.weights))
            return lie.pose_from_state(_rot_from_quat(self.particles_q[idx]), self.particles_p[idx])
        p = np.average(self.particles_p, axis=0, weights=self.weights)
        q = self.particles_q[int(np.argmax(self.weights))]
        return lie.pose_from_state(_rot_from_quat(q), p)

    def _state_matrix(self) -> np.ndarray:
        out = np.zeros((self.num_particles, self.dim), dtype=float)
        out[:, 0:3] = self.particles_p
        out[:, 3:6] = self.particles_v
        out[:, 9:12] = self.particles_bg
        out[:, 12:15] = self.particles_ba
        return out

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
        scale = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * scale, (R[2, 1] - R[1, 2]) / scale, (R[0, 2] - R[2, 0]) / scale, (R[1, 0] - R[0, 1]) / scale])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        scale = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / scale, 0.25 * scale, (R[0, 1] + R[1, 0]) / scale, (R[0, 2] + R[2, 0]) / scale])
    elif R[1, 1] > R[2, 2]:
        scale = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / scale, (R[0, 1] + R[1, 0]) / scale, 0.25 * scale, (R[1, 2] + R[2, 1]) / scale])
    else:
        scale = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / scale, (R[0, 2] + R[2, 0]) / scale, (R[1, 2] + R[2, 1]) / scale, 0.25 * scale])
    return _quat_normalize(q)

def _quat_normalize_batch(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    bad = ~np.isfinite(norm[:, 0]) | (norm[:, 0] <= 0.0)
    out = q / np.where(bad[:, None], 1.0, norm)
    out[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    sign = np.where(out[:, :1] < 0.0, -1.0, 1.0)
    return out * sign


def _quat_multiply_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = _quat_normalize_batch(q1)
    q2 = _quat_normalize_batch(q2)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return _quat_normalize_batch(out)


def _quat_from_rotvec_batch(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(-1, 3)
    angle = np.linalg.norm(phi, axis=1)
    out = np.empty((phi.shape[0], 4), dtype=float)
    small = angle < 1.0e-12
    out[small, 0] = 1.0
    out[small, 1:4] = 0.5 * phi[small]
    half = 0.5 * angle[~small]
    axis = phi[~small] / angle[~small, None]
    out[~small, 0] = np.cos(half)
    out[~small, 1:4] = axis * np.sin(half)[:, None]
    return _quat_normalize_batch(out)


def _rot_from_quat_batch(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize_batch(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def _skew_batch(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    out = np.zeros((vectors.shape[0], 3, 3), dtype=float)
    out[:, 0, 1] = -vectors[:, 2]
    out[:, 0, 2] = vectors[:, 1]
    out[:, 1, 0] = vectors[:, 2]
    out[:, 1, 2] = -vectors[:, 0]
    out[:, 2, 0] = -vectors[:, 1]
    out[:, 2, 1] = vectors[:, 0]
    return out


def _left_jacobian_so3_batch(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(-1, 3)
    n = phi.shape[0]
    K = _skew_batch(phi)
    K2 = np.einsum("nij,njk->nik", K, K)
    theta = np.linalg.norm(phi, axis=1)
    I = np.broadcast_to(np.eye(3, dtype=float), (n, 3, 3)).copy()
    A = np.empty(n, dtype=float)
    B = np.empty(n, dtype=float)
    small = theta < 1e-8
    th2 = theta * theta
    A[small] = 0.5 - th2[small] / 24.0 + th2[small] * th2[small] / 720.0
    B[small] = 1.0 / 6.0 - th2[small] / 120.0 + th2[small] * th2[small] / 5040.0
    A[~small] = (1.0 - np.cos(theta[~small])) / th2[~small]
    B[~small] = (theta[~small] - np.sin(theta[~small])) / (theta[~small] ** 3)
    return I + A[:, None, None] * K + B[:, None, None] * K2


def _gamma2_so3_batch(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(-1, 3)
    n = phi.shape[0]
    K = _skew_batch(phi)
    K2 = np.einsum("nij,njk->nik", K, K)
    theta = np.linalg.norm(phi, axis=1)
    I = np.broadcast_to(0.5 * np.eye(3, dtype=float), (n, 3, 3)).copy()
    A = np.empty(n, dtype=float)
    B = np.empty(n, dtype=float)
    small = theta < 1e-8
    th2 = theta * theta
    A[small] = 1.0 / 6.0 - th2[small] / 120.0 + th2[small] * th2[small] / 5040.0
    B[small] = 1.0 / 24.0 - th2[small] / 720.0 + th2[small] * th2[small] / 40320.0
    A[~small] = (theta[~small] - np.sin(theta[~small])) / (theta[~small] ** 3)
    B[~small] = (theta[~small] ** 2 + 2.0 * np.cos(theta[~small]) - 2.0) / (2.0 * theta[~small] ** 4)
    return I + A[:, None, None] * K + B[:, None, None] * K2

