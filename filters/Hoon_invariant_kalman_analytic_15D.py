from __future__ import annotations

import numpy as np

from filters.Hoon_invariant_kalman_filter_15D import HoonInvariantKalmanFilter15D, _finite_difference_jacobian
from models import Hoon_invariant_inekf as lie
from models.Hoon_lie_group_utils import exp_so3, gamma2_so3, hat_so3, left_jacobian_so3, log_so3, plus_right


class HoonInvariantKalmanAnalytic15D(HoonInvariantKalmanFilter15D):
    """Analytic left-invariant-error EKF matching ghaggin/invariant-ekf.

    State error order is ``[dR, dv, dp, dbg, dba]``. Perturbations and
    corrections use ``X @ Exp(delta)``, as in the reference LIEKF.
    """

    def __init__(self, *args, **kwargs) -> None:
        motion_config = kwargs.get("motion_config") or {}
        self.jacobian_mode = str(motion_config.get("jacobian_mode", "analytic")).lower()
        if self.jacobian_mode not in {"finite", "analytic"}:
            raise ValueError("jacobian_mode must be 'finite' or 'analytic'.")
        super().__init__(*args, **kwargs)

    @classmethod
    def from_configs(cls, dataset_config: dict, compare_config: dict) -> "HoonInvariantKalmanAnalytic15D":
        cfg = compare_config.get(
            "Hoon_invariant_kalman_analytic_15d",
            compare_config.get("Hoon_invariant_kalman_filter_15d", compare_config.get("invariant_kalman_filter_15d", compare_config)),
        )
        return cls(
            pose_type=dataset_config.get("pose_type", cfg.get("pose_type", "3d")),
            mode=dataset_config.get("mode", cfg.get("mode", "fused")),
            motion_config=cfg.get("motion_model", {}),
            measurement_config=cfg.get("measurement_model", {}),
            initialization_config=cfg.get("initialization", {}),
        )

    def _process_jacobian(
        self,
        X_prev: np.ndarray,
        gyro_bias_prev: np.ndarray,
        accel_bias_prev: np.ndarray,
        control: np.ndarray,
        dt: float,
        X_pred: np.ndarray,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        if self.jacobian_mode == "finite":
            return super()._process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=eps)
        return self._analytic_process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt)

    def _position_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        if getattr(self, "jacobian_mode", "finite") == "finite":
            return super()._position_measurement_jacobian(eps=eps)
        H = np.zeros((3, self.error_dim), dtype=float)
        # X Exp(delta) gives p_plus ~= p + R dp.
        H[:, 6:9] = self.Rot
        return H

    def _velocity_measurement_jacobian(self, eps: float = 1.0e-6) -> np.ndarray:
        if getattr(self, "jacobian_mode", "finite") == "finite":
            return super()._velocity_measurement_jacobian(eps=eps)
        H = np.zeros((3, self.error_dim), dtype=float)
        # X Exp(delta) gives v_plus ~= v + R dv.
        H[:, 3:6] = self.Rot
        return H

    def finite_process_jacobian_for_validation(self, X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=1e-6):
        return super()._process_jacobian(X_prev, gyro_bias_prev, accel_bias_prev, control, dt, X_pred, eps=eps)

    def finite_position_jacobian_for_validation(self, eps=1e-6):
        return super()._position_measurement_jacobian(eps=eps)

    def finite_velocity_jacobian_for_validation(self, eps=1e-6):
        return super()._velocity_measurement_jacobian(eps=eps)

    def _analytic_process_jacobian(
        self,
        X_prev: np.ndarray,
        gyro_bias_prev: np.ndarray,
        accel_bias_prev: np.ndarray,
        control: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        Rot, _velocity, _position = lie.from_matrix(X_prev)
        u = np.asarray(control, dtype=float).reshape(-1)
        dt = max(float(dt), 0.0)
        dt2 = dt * dt
        use_accel_bias = self.translation_input_type == "acceleration"
        use_gyro_bias = self.use_imu_rotation and self.rotation_input_type == "rate"
        w = u[3:6] - gyro_bias_prev if use_gyro_bias else u[3:6]
        a = u[0:3] - accel_bias_prev if use_accel_bias else u[0:3]
        phi = w * dt if self.rotation_input_type == "rate" else w
        E = exp_so3(phi)
        A = E.T
        G1 = left_jacobian_so3(phi)
        G2 = gamma2_so3(phi)
        acc_v = G1 @ a * dt
        acc_p = G2 @ a * dt2

        Phi = np.eye(self.error_dim, dtype=float)
        Phi[:9, :9] = 0.0
        Phi[0:3, 0:3] = A
        if self.use_imu_velocity and self.translation_input_frame == "body" and self.translation_input_type == "acceleration":
            Phi[3:6, 0:3] = -A @ hat_so3(acc_v)
            Phi[3:6, 3:6] = A
            Phi[6:9, 0:3] = -A @ hat_so3(acc_p)
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A
            Phi[3:6, 12:15] = -A @ G1 * dt
            Phi[6:9, 12:15] = -A @ G2 * dt2
        elif self.use_imu_velocity and self.translation_input_frame == "world" and self.translation_input_type == "acceleration":
            Phi[3:6, 3:6] = A
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A
            Phi[3:6, 12:15] = -A * dt
            Phi[6:9, 12:15] = -0.5 * A * dt2
        else:
            Phi[3:6, 3:6] = A
            Phi[6:9, 3:6] = A * dt
            Phi[6:9, 6:9] = A

        gyro_bias_is_uncertain = (
            self.update_biases
            or np.any(self.P[9:12, :])
            or np.any(self.P[:, 9:12])
            or np.any(self.process_noise_diag[9:12])
        )
        if use_gyro_bias and gyro_bias_is_uncertain:
            dtheta_dbg, dv_dbg, dp_dbg = self._gyro_bias_coupling_blocks(phi, a, dt)
            Phi[0:3, 9:12] = dtheta_dbg
            if self.use_imu_velocity and self.translation_input_frame == "body" and self.translation_input_type == "acceleration":
                Phi[3:6, 9:12] = dv_dbg
                Phi[6:9, 9:12] = dp_dbg
        Phi[9:12, 9:12] = np.eye(3)
        Phi[12:15, 12:15] = np.eye(3)
        return np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

    def _gyro_bias_coupling_blocks(
        self,
        phi: np.ndarray,
        accel_body: np.ndarray,
        dt: float,
        eps: float = 1e-7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discrete local derivatives with respect to the fixed gyro bias."""
        phi = np.asarray(phi, dtype=float).reshape(3)
        accel_body = np.asarray(accel_body, dtype=float).reshape(3)
        E = exp_so3(phi)
        A = E.T
        base_v = left_jacobian_so3(phi) @ accel_body * dt
        base_p = gamma2_so3(phi) @ accel_body * dt * dt
        dtheta_dbg = np.zeros((3, 3), dtype=float)
        dv_dbg = np.zeros((3, 3), dtype=float)
        dp_dbg = np.zeros((3, 3), dtype=float)
        for col in range(3):
            step = np.zeros(3, dtype=float)
            step[col] = eps
            phi_p = phi - step * dt
            phi_m = phi + step * dt
            dtheta_p = log_so3(A @ exp_so3(phi_p))
            dtheta_m = log_so3(A @ exp_so3(phi_m))
            v_p = A @ (left_jacobian_so3(phi_p) @ accel_body * dt - base_v)
            v_m = A @ (left_jacobian_so3(phi_m) @ accel_body * dt - base_v)
            p_p = A @ (gamma2_so3(phi_p) @ accel_body * dt * dt - base_p)
            p_m = A @ (gamma2_so3(phi_m) @ accel_body * dt * dt - base_p)
            dtheta_dbg[:, col] = (dtheta_p - dtheta_m) / (2.0 * eps)
            dv_dbg[:, col] = (v_p - v_m) / (2.0 * eps)
            dp_dbg[:, col] = (p_p - p_m) / (2.0 * eps)
        return dtheta_dbg, dv_dbg, dp_dbg


def analytic_position_measurement_jacobian_from_state(
    Rot: np.ndarray,
    error_dim: int = 15,
) -> np.ndarray:
    H = np.zeros((3, error_dim), dtype=float)
    H[:, 6:9] = np.asarray(Rot, dtype=float).reshape(3, 3)
    return H


def finite_position_measurement_jacobian_from_state(X: np.ndarray, error_dim: int = 15, eps: float = 1e-6) -> np.ndarray:
    base = lie.from_matrix(X)[2]

    def measurement(delta: np.ndarray) -> np.ndarray:
        _Rot, _v, p = lie.from_matrix(plus_right(X, delta[:9]))
        return p - base

    return _finite_difference_jacobian(measurement, error_dim, eps)
