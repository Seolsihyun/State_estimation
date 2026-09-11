"""Lie group primitives used by the InEKF implementation.

Conventions follow the micro Lie theory notation used for state estimation:

- SO(3) tangent vectors are 3-vectors.
- SE_2(3) tangent vectors are ordered as ``[phi, rho_v, rho_p]``.
- ``plus_right(X, tau)`` is ``X @ Exp(tau)``.
- ``minus_right(Y, X)`` is ``Log(X^{-1} @ Y)``.

The elementary composition/inverse Jacobians below are written for right
perturbations. The Exp/Log/plus/minus Jacobian helpers intentionally use
central finite differences in tangent coordinates; this keeps the API aligned
with the paper's Jacobian blocks while avoiding a partial closed-form SE_2(3)
implementation that would be easier to misuse.
"""

from __future__ import annotations

import numpy as np


_EPS = 1.0e-12
_JAC_EPS = 1.0e-7


def _as_vector(value: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(size)


def hat_so3(w: np.ndarray) -> np.ndarray:
    x, y, z = _as_vector(w, 3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def vee_so3(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float).reshape(3, 3)
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


def exp_so3(phi: np.ndarray) -> np.ndarray:
    phi = _as_vector(phi, 3)
    theta = float(np.linalg.norm(phi))
    K = hat_so3(phi)
    K2 = K @ K
    if theta < 1.0e-8:
        return np.eye(3) + K + 0.5 * K2
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / theta**2) * K2


def log_so3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1.0e-8:
        return vee_so3(0.5 * (R - R.T))
    if np.pi - theta < 1.0e-5:
        axis = np.empty(3, dtype=float)
        diag = np.diag(R)
        idx = int(np.argmax(diag))
        axis[idx] = np.sqrt(max(diag[idx] - cos_theta, 0.0) / (1.0 - cos_theta))
        j = (idx + 1) % 3
        k = (idx + 2) % 3
        denom = max(axis[idx] * (1.0 - cos_theta), _EPS)
        axis[j] = (R[j, idx] + R[idx, j]) * 0.5 / denom
        axis[k] = (R[k, idx] + R[idx, k]) * 0.5 / denom
        axis = axis / max(float(np.linalg.norm(axis)), _EPS)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee_so3(R - R.T)


def left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    phi = _as_vector(phi, 3)
    theta = float(np.linalg.norm(phi))
    K = hat_so3(phi)
    K2 = K @ K
    if theta < 1.0e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * K2
    return np.eye(3) + ((1.0 - np.cos(theta)) / theta**2) * K + ((theta - np.sin(theta)) / theta**3) * K2


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    return left_jacobian_so3(-_as_vector(phi, 3))


def left_jacobian_inv_so3(phi: np.ndarray) -> np.ndarray:
    phi = _as_vector(phi, 3)
    theta = float(np.linalg.norm(phi))
    K = hat_so3(phi)
    K2 = K @ K
    if theta < 1.0e-8:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * K2
    scale = (1.0 / theta**2) - ((1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
    return np.eye(3) - 0.5 * K + scale * K2


def right_jacobian_inv_so3(phi: np.ndarray) -> np.ndarray:
    return left_jacobian_inv_so3(-_as_vector(phi, 3))


def gamma2_so3(phi: np.ndarray) -> np.ndarray:
    phi = _as_vector(phi, 3)
    theta = float(np.linalg.norm(phi))
    K = hat_so3(phi)
    K2 = K @ K
    if theta < 1.0e-8:
        return 0.5 * np.eye(3) + (1.0 / 6.0) * K + (1.0 / 24.0) * K2
    return (
        0.5 * np.eye(3)
        + ((theta - np.sin(theta)) / theta**3) * K
        + ((theta**2 + 2.0 * np.cos(theta) - 2.0) / (2.0 * theta**4)) * K2
    )


def hat_se23(xi: np.ndarray) -> np.ndarray:
    """Return se_2(3) hat matrix for xi = [phi, rho_v, rho_p]."""
    xi = _as_vector(xi, 9)
    Xi = np.zeros((5, 5), dtype=float)
    Xi[:3, :3] = hat_so3(xi[0:3])
    Xi[:3, 3] = xi[3:6]
    Xi[:3, 4] = xi[6:9]
    return Xi


def vee_se23(Xi_hat: np.ndarray) -> np.ndarray:
    Xi_hat = np.asarray(Xi_hat, dtype=float).reshape(5, 5)
    return np.concatenate([vee_so3(Xi_hat[:3, :3]), Xi_hat[:3, 3], Xi_hat[:3, 4]])


def as_matrix(Rot: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    X = np.eye(5, dtype=float)
    X[:3, :3] = np.asarray(Rot, dtype=float).reshape(3, 3)
    X[:3, 3] = _as_vector(v, 3)
    X[:3, 4] = _as_vector(p, 3)
    return X


def from_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    return X[:3, :3].copy(), X[:3, 3].copy(), X[:3, 4].copy()


def exp_se23(xi: np.ndarray) -> np.ndarray:
    """Exponential map for xi = [phi, rho_v, rho_p]."""
    xi = _as_vector(xi, 9)
    R = exp_so3(xi[0:3])
    J = left_jacobian_so3(xi[0:3])
    return as_matrix(R, J @ xi[3:6], J @ xi[6:9])


def log_se23(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    phi = log_so3(X[:3, :3])
    J_inv = left_jacobian_inv_so3(phi)
    return np.concatenate([phi, J_inv @ X[:3, 3], J_inv @ X[:3, 4]])


def adjoint_se23(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    R = X[:3, :3]
    v = X[:3, 3]
    p = X[:3, 4]
    Ad = np.zeros((9, 9), dtype=float)
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = hat_so3(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = hat_so3(p) @ R
    Ad[6:9, 6:9] = R
    return Ad


def compose(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).reshape(5, 5) @ np.asarray(Y, dtype=float).reshape(5, 5)


def inverse(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    R_T = X[:3, :3].T
    return as_matrix(R_T, -R_T @ X[:3, 3], -R_T @ X[:3, 4])


def plus_right(X: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus operator: X (+) tau = X @ Exp(tau)."""
    return compose(X, exp_se23(tau))


def minus_right(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Right minus operator: Y (-) X = Log(X^{-1} @ Y)."""
    return log_se23(compose(inverse(X), Y))


def correction_left(X: np.ndarray, delta_xi: np.ndarray) -> np.ndarray:
    """Left correction injection used by the current InEKF update."""
    return compose(exp_se23(delta_xi), X)


def correction_right(X: np.ndarray, delta_xi: np.ndarray) -> np.ndarray:
    """Right correction injection, equivalent to plus_right(X, delta_xi)."""
    return plus_right(X, delta_xi)


def symmetrize_covariance(P: np.ndarray, floor: float | None = None, ceiling: float | None = None) -> np.ndarray:
    P = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    P = 0.5 * (P + P.T)
    if floor is None and ceiling is None:
        return P
    try:
        eigvals, eigvecs = np.linalg.eigh(P)
        if floor is not None:
            eigvals = np.maximum(eigvals, float(floor))
        if ceiling is not None:
            eigvals = np.minimum(eigvals, float(ceiling))
        P = (eigvecs * eigvals) @ eigvecs.T
        return 0.5 * (P + P.T)
    except np.linalg.LinAlgError:
        diag = np.diag(P).copy()
        if floor is not None:
            diag = np.maximum(diag, float(floor))
        if ceiling is not None:
            diag = np.minimum(diag, float(ceiling))
        return np.diag(diag)


def jacobian_inverse(X: np.ndarray) -> np.ndarray:
    """Right-perturbation Jacobian of Inv(X): d Inv(X) = -Ad_X dx."""
    return -adjoint_se23(X)


def jacobian_composition_wrt_first(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Right-perturbation Jacobian of X * Y with respect to X."""
    del X
    return adjoint_se23(inverse(Y))


def jacobian_composition_wrt_second(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Right-perturbation Jacobian of X * Y with respect to Y."""
    del X, Y
    return np.eye(9, dtype=float)


def jacobian_exp(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SE_2(3) Exp, computed by central differences."""
    xi = _as_vector(xi, 9)
    X = exp_se23(xi)

    def residual(delta: np.ndarray) -> np.ndarray:
        return minus_right(exp_se23(xi + delta), X)

    return _finite_difference_vector(residual, 9)


def jacobian_log(X: np.ndarray) -> np.ndarray:
    """Jacobian of Log(X) for right perturbations, computed numerically."""
    X = np.asarray(X, dtype=float).reshape(5, 5)
    base = log_se23(X)

    def residual(delta: np.ndarray) -> np.ndarray:
        return log_se23(plus_right(X, delta)) - base

    return _finite_difference_vector(residual, 9)


def jacobian_plus_wrt_state(X: np.ndarray, tau: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    tau = _as_vector(tau, 9)
    Y = plus_right(X, tau)

    def residual(delta: np.ndarray) -> np.ndarray:
        return minus_right(plus_right(plus_right(X, delta), tau), Y)

    return _finite_difference_vector(residual, 9)


def jacobian_plus_wrt_tau(X: np.ndarray, tau: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(5, 5)
    tau = _as_vector(tau, 9)
    Y = plus_right(X, tau)

    def residual(delta: np.ndarray) -> np.ndarray:
        return minus_right(plus_right(X, tau + delta), Y)

    return _finite_difference_vector(residual, 9)


def jacobian_minus_wrt_first(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float).reshape(5, 5)
    X = np.asarray(X, dtype=float).reshape(5, 5)
    base = minus_right(Y, X)

    def residual(delta: np.ndarray) -> np.ndarray:
        return minus_right(plus_right(Y, delta), X) - base

    return _finite_difference_vector(residual, 9)


def jacobian_minus_wrt_second(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float).reshape(5, 5)
    X = np.asarray(X, dtype=float).reshape(5, 5)
    base = minus_right(Y, X)

    def residual(delta: np.ndarray) -> np.ndarray:
        return minus_right(Y, plus_right(X, delta)) - base

    return _finite_difference_vector(residual, 9)


def _finite_difference_vector(fn, dim: int, eps: float = _JAC_EPS) -> np.ndarray:
    J = np.zeros((dim, dim), dtype=float)
    for col in range(dim):
        step = np.zeros(dim, dtype=float)
        step[col] = eps
        J[:, col] = (fn(step) - fn(-step)) / (2.0 * eps)
    return J
