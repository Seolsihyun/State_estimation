"""Create fair comparison figures for pure-IMU DR and learned velocity + InEKF."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
PURE_ROOT = ROOT / "validation/results/cf231_run5_pure_imu_improved"
LEARNED_ROOT = ROOT / "validation/results/cf231_leave5_learned_velocity"
OUTPUT = ROOT / "validation/results/cf231_pure_vs_learned"


def load() -> dict[str, object]:
    pure = np.load(PURE_ROOT / "trajectories.npz")
    learned = np.load(LEARNED_ROOT / "trajectories.npz")
    if not np.allclose(pure["time"], learned["time"]):
        raise ValueError("The two experiments do not share timestamps.")
    if not np.allclose(pure["gt_position"], learned["gt_position"]):
        raise ValueError("The two experiments do not share GT positions.")
    return {
        "time": pure["time"],
        "gt_position": pure["gt_position"],
        "gt_rotation": pure["gt_rotation"],
        "pure_position": pure["improved_position"],
        "pure_rotation": pure["improved_rotation"],
        "learned_position": learned["learned_position"],
        "learned_rotation": learned["learned_body_rotation"],
        "pure_summary": json.loads((PURE_ROOT / "summary.json").read_text()),
        "learned_summary": json.loads((LEARNED_ROOT / "summary.json").read_text()),
    }


def position_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(estimate - truth, axis=1)


def attitude_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", truth, estimate)
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def plot_trajectory(data: dict[str, object], figures: Path) -> None:
    time = data["time"]
    gt = data["gt_position"]
    pure = data["pure_position"]
    learned = data["learned_position"]
    pure_error = position_error(pure, gt)
    colors = {"pure": "tab:orange", "learned": "tab:blue"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for axis, horizon in zip(axes[:2], (10, 20)):
        mask = time <= horizon
        axis.plot(gt[mask, 0], gt[mask, 1], "k", lw=2.1, label="GT")
        axis.plot(
            pure[mask, 0], pure[mask, 1], color=colors["pure"], lw=1.6,
            label="pure IMU DR",
        )
        axis.plot(
            learned[mask, 0], learned[mask, 1], color=colors["learned"], lw=1.6,
            label="learned velocity + InEKF",
        )
        axis.set_title(f"First {horizon} s")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    axis = axes[2]
    axis.plot(gt[:, 0], gt[:, 1], "k", lw=2.1, label="GT full")
    axis.plot(
        learned[:, 0], learned[:, 1], color=colors["learned"], lw=1.0,
        label="learned full",
    )
    crossing = np.flatnonzero(pure_error >= 10.0)
    cutoff = int(crossing[0]) if crossing.size else len(time) - 1
    axis.plot(
        pure[: cutoff + 1, 0], pure[: cutoff + 1, 1],
        color=colors["pure"], lw=1.5,
        label=f"pure until 10 m ({time[cutoff]:.1f} s)",
    )
    margin_x = max(0.2, 0.15 * np.ptp(gt[:, 0]))
    margin_y = max(0.2, 0.15 * np.ptp(gt[:, 1]))
    axis.set_xlim(gt[:, 0].min() - margin_x, gt[:, 0].max() + margin_x)
    axis.set_ylim(gt[:, 1].min() - margin_y, gt[:, 1].max() + margin_y)
    axis.set_title("Full duration at GT scale")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Held-out Run 5 trajectory: two IMU-input methods")
    fig.tight_layout()
    fig.savefig(figures / "01_trajectory_comparison.png", dpi=200)
    plt.close(fig)


def plot_errors(data: dict[str, object], figures: Path) -> None:
    time = data["time"]
    gt_position = data["gt_position"]
    gt_rotation = data["gt_rotation"]
    pure_position_error = position_error(data["pure_position"], gt_position)
    learned_position_error = position_error(data["learned_position"], gt_position)
    pure_attitude_error = attitude_error(data["pure_rotation"], gt_rotation)
    learned_attitude_error = attitude_error(data["learned_rotation"], gt_rotation)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].semilogy(
        time, np.maximum(pure_position_error, 1e-6), color="tab:orange",
        lw=1.4, label="pure IMU DR",
    )
    axes[0].semilogy(
        time, np.maximum(learned_position_error, 1e-6), color="tab:blue",
        lw=1.4, label="learned velocity + InEKF",
    )
    axes[1].plot(time, pure_attitude_error, color="tab:orange", lw=1.0, label="pure IMU DR")
    axes[1].plot(
        time, learned_attitude_error, color="tab:blue", lw=1.0,
        label="learned velocity + InEKF",
    )
    axes[0].set_ylabel("3D position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.suptitle("Error over the full 181.3 s sequence")
    fig.tight_layout()
    fig.savefig(figures / "02_error_comparison.png", dpi=200)
    plt.close(fig)


def plot_rpy(data: dict[str, object], figures: Path) -> None:
    time = data["time"]
    gt = Rotation.from_matrix(data["gt_rotation"]).as_euler("xyz", degrees=True)
    pure = Rotation.from_matrix(data["pure_rotation"]).as_euler("xyz", degrees=True)
    learned = Rotation.from_matrix(data["learned_rotation"]).as_euler("xyz", degrees=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for index, label in enumerate(("roll", "pitch", "yaw")):
        axes[index].plot(time, gt[:, index], "k", lw=1.2, label="GT")
        axes[index].plot(time, pure[:, index], color="tab:orange", lw=0.8, label="pure IMU DR")
        axes[index].plot(
            time, learned[:, index], color="tab:blue", lw=0.8,
            label="learned velocity + InEKF",
        )
        axes[index].set_ylabel(f"{label} [deg]")
        axes[index].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("RPY comparison")
    fig.tight_layout()
    fig.savefig(figures / "03_rpy_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = load()
    plot_trajectory(data, figures)
    plot_errors(data, figures)
    plot_rpy(data, figures)
    print(f"Created comparison figures in {figures}")


if __name__ == "__main__":
    main()
