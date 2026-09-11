"""Small-TCN learned IMU odometry with fixed-bias InEKF attitude.

The network is intentionally compact: one input convolution, three residual
temporal blocks and a small regression head.  Runs 3/4/9/10 are used for
supervision and Run 5 is strictly held out.  Run-5 GT is accessed only after
the IMU-only trajectory has been produced.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader, Dataset

from validation.cf231_loader import load_cf231_run, synchronize_to_imu
from validation.run_cf231_learned_velocity_leave5 import (
    heading_velocity,
    parse_run_ids,
    training_quality_mask,
)
from validation.run_cf231_sensor_frame_leave5 import (
    Trajectory,
    calculate_metrics,
    estimate_sensor_calibration,
)
from validation.run_cf231_shallow_learned_velocity_inekf import (
    fixed_bias_attitude,
    integrate_learned_velocity,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SequenceSet:
    windows: np.ndarray
    target: np.ndarray
    end_indices: np.ndarray


class WindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        target: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        augment: bool,
    ) -> None:
        self.windows = windows
        self.target = target
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.augment = augment

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int):
        window = self.windows[index].copy()
        if self.augment:
            # Small constant sensor offsets and white noise discourage the
            # network from identifying a recording by its session bias.
            window[:, 0:3] += np.random.normal(0.0, 0.015, (1, 3))
            window[:, 3:6] += np.random.normal(0.0, 2.0e-4, (1, 3))
            window[:, 0:3] += np.random.normal(0.0, 0.01, window[:, 0:3].shape)
            window[:, 3:6] += np.random.normal(0.0, 1.0e-4, window[:, 3:6].shape)
        normalized = (window - self.input_mean) / self.input_std
        target = (self.target[index] - self.target_mean) / self.target_std
        return (
            torch.from_numpy(normalized.T.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                dilation=dilation,
                padding=padding,
            ),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                dilation=dilation,
                padding=padding,
            ),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class SmallTCN(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, channels, kernel_size=7, padding=3),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(channels, 1),
            TemporalBlock(channels, 2),
            TemporalBlock(channels, 4),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * channels, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 3),
        )

    def forward(self, value):
        feature = self.blocks(self.stem(value))
        pooled = torch.cat((feature.mean(dim=-1), feature[:, :, -1]), dim=1)
        return self.head(pooled)


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
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--training-stride", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=24)
    parser.add_argument("--final-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--static-seconds", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation/results/cf231_leave5_small_tcn_velocity_inekf",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def corrected_imu(run) -> np.ndarray:
    imu = run.imu.copy()
    initial = min(100, imu.shape[0])
    imu[:, 3:6] -= np.mean(imu[:initial, 3:6], axis=0)
    return imu


def make_sequences(run, window: int, downsample: int, stride: int) -> SequenceSet:
    quality = training_quality_mask(run)
    target = heading_velocity(run)
    imu = corrected_imu(run)
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []
    for end in range(window - 1, run.time.size, stride):
        start = end - window + 1
        if not (
            quality[end]
            and np.mean(quality[start : end + 1]) >= 0.90
            and np.all(np.isfinite(target[end]))
        ):
            continue
        windows.append(imu[start : end + 1 : downsample])
        targets.append(target[end])
        indices.append(end)
    return SequenceSet(
        windows=np.asarray(windows, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float32),
        end_indices=np.asarray(indices, dtype=np.int64),
    )


def inference_sequences(run, window: int, downsample: int) -> SequenceSet:
    imu = corrected_imu(run)
    indices = np.arange(window - 1, run.time.size, dtype=np.int64)
    windows = np.asarray(
        [imu[end - window + 1 : end + 1 : downsample] for end in indices],
        dtype=np.float32,
    )
    return SequenceSet(
        windows=windows,
        target=np.empty((indices.size, 3), dtype=np.float32),
        end_indices=indices,
    )


def normalization(sets: list[SequenceSet]):
    windows = np.concatenate([item.windows for item in sets], axis=0)
    targets = np.concatenate([item.target for item in sets], axis=0)
    input_mean = windows.mean(axis=(0, 1))
    input_std = np.maximum(windows.std(axis=(0, 1)), 1.0e-5)
    target_mean = targets.mean(axis=0)
    target_std = np.maximum(targets.std(axis=0), 1.0e-4)
    return input_mean, input_std, target_mean, target_std


def fit_model(
    sets: list[SequenceSet],
    channels: int,
    epochs: int,
    batch_size: int,
    seed: int,
):
    set_seed(seed)
    stats = normalization(sets)
    windows = np.concatenate([item.windows for item in sets], axis=0)
    targets = np.concatenate([item.target for item in sets], axis=0)
    dataset = WindowDataset(windows, targets, *stats, augment=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = SmallTCN(channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for inputs, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            point_loss = nn.functional.smooth_l1_loss(prediction, target)
            mean_loss = torch.mean(torch.square(torch.mean(prediction - target, dim=0)))
            loss = point_loss + 0.15 * mean_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * inputs.shape[0]
            count += inputs.shape[0]
        scheduler.step()
        loss_history.append(total / max(count, 1))
    return model, stats, loss_history


def predict(model, stats, windows: np.ndarray, batch_size: int) -> np.ndarray:
    input_mean, input_std, target_mean, target_std = stats
    dummy = np.zeros((windows.shape[0], 3), dtype=np.float32)
    dataset = WindowDataset(
        windows,
        dummy,
        input_mean,
        input_std,
        target_mean,
        target_std,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            values.append(model(inputs).cpu().numpy())
    normalized = np.vstack(values)
    return normalized * target_std + target_mean


def integrate_with_gt_yaw(run, indices, heading_velocity_prediction):
    all_indices = np.arange(run.time.size)
    full_prediction = np.column_stack(
        [
            np.interp(all_indices, indices, heading_velocity_prediction[:, axis])
            for axis in range(3)
        ]
    )
    yaw = run.rotation.as_euler("xyz")[:, 2]
    angle = yaw
    cosine = np.cos(angle)
    sine = np.sin(angle)
    world = np.column_stack(
        (
            cosine * full_prediction[:, 0] - sine * full_prediction[:, 1],
            sine * full_prediction[:, 0] + cosine * full_prediction[:, 1],
            full_prediction[:, 2],
        )
    )
    velocity = world
    position = np.empty_like(run.position)
    position[0] = run.position[0]
    for index in range(1, run.time.size):
        dt = float(run.time[index] - run.time[index - 1])
        position[index] = position[index - 1] + 0.5 * (
            velocity[index - 1] + velocity[index]
        ) * dt
    error = np.linalg.norm(position - run.position, axis=1)
    return float(np.sqrt(np.mean(np.square(error)))), float(error[-1])


def cross_validate(runs, sets, args):
    per_run = {}
    residuals = []
    best_epochs = []
    for held_out in sorted(runs):
        train_ids = [run_id for run_id in sorted(runs) if run_id != held_out]
        model, stats, history = fit_model(
            [sets[run_id] for run_id in train_ids],
            args.channels,
            args.cv_epochs,
            args.batch_size,
            args.seed + held_out,
        )
        prediction = predict(
            model,
            stats,
            sets[held_out].windows,
            args.batch_size,
        )
        residual = prediction - sets[held_out].target
        residuals.append(residual)
        trajectory_rmse, final_error = integrate_with_gt_yaw(
            runs[held_out],
            sets[held_out].end_indices,
            prediction,
        )
        per_run[str(held_out)] = {
            "samples": int(prediction.shape[0]),
            "velocity_rmse_m_s": np.sqrt(np.mean(np.square(residual), axis=0)).tolist(),
            "velocity_bias_m_s": np.mean(residual, axis=0).tolist(),
            "trajectory_rmse_m": trajectory_rmse,
            "trajectory_final_m": final_error,
            "final_training_loss": history[-1],
        }
        best_epochs.append(args.cv_epochs)
        print(f"CV run {held_out}: {per_run[str(held_out)]}", flush=True)
    pooled = np.vstack(residuals)
    return (
        per_run,
        np.mean(pooled, axis=0),
        np.mean(np.square(pooled), axis=0),
        int(np.median(best_epochs)),
    )


def load_baselines():
    output = ROOT / "validation/results/cf231_leave5_shallow_learned_velocity_inekf"
    data = np.load(output / "trajectories.npz")
    zeros = np.zeros_like(data["pure_position"])
    pure = Trajectory(
        time=data["time"],
        imu_rotation=data["pure_rotation"],
        body_rotation=data["pure_rotation"],
        velocity=np.zeros_like(data["pure_position"]),
        position=data["pure_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )
    shallow = Trajectory(
        time=data["time"],
        imu_rotation=data["shallow_rotation"],
        body_rotation=data["shallow_rotation"],
        velocity=data["shallow_velocity"],
        position=data["shallow_position"],
        position_std=zeros,
        attitude_std_deg=zeros,
    )
    return pure, shallow


def plot_results(output: Path, run, shallow, tcn) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = {
        "Shallow learned velocity + InEKF": (shallow, "tab:orange"),
        "Small TCN + InEKF": (tcn, "tab:blue"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for axis, horizon in zip(axes, (20.0, 60.0, run.time[-1])):
        mask = run.time <= horizon
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2, label="GT")
        for name, (trajectory, color) in methods.items():
            axis.plot(
                trajectory.position[mask, 0],
                trajectory.position[mask, 1],
                color=color,
                lw=1.15,
                label=name,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend(fontsize=8)
    axes[0].set_title("0–20 s")
    axes[1].set_title("0–60 s")
    axes[2].set_title("Full held-out Run 5")
    fig.tight_layout()
    fig.savefig(figures / "01_trajectory_comparison.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, (trajectory, color) in methods.items():
        metrics, position_error, attitude_error = calculate_metrics(trajectory, run)
        axes[0].semilogy(
            run.time,
            np.maximum(position_error, 1.0e-5),
            color=color,
            label=f"{name} ({metrics['position_rmse_m']:.2f} m RMSE)",
        )
        axes[1].plot(run.time, attitude_error, color=color, label=name)
    axes[0].set_ylabel("3D position error [m]")
    axes[1].set_ylabel("SO(3) error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "02_error_comparison.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, horizon in zip(axes, (60.0, run.time[-1])):
        mask = run.time <= horizon
        axis.plot(run.position[:, 0], run.position[:, 1], "k", lw=2.2, label="GT")
        for name, (trajectory, color) in methods.items():
            axis.plot(
                trajectory.position[mask, 0],
                trajectory.position[mask, 1],
                color=color,
                lw=1.15,
                label=name,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend(fontsize=8)
    axes[0].set_title("First 60 s")
    axes[1].set_title("Full held-out Run 5")
    fig.tight_layout()
    fig.savefig(figures / "03_learned_trajectory_comparison.png", dpi=190)
    plt.close(fig)

    gt_rpy = run.rotation.as_euler("xyz", degrees=True)
    estimated_rpy = Rotation.from_matrix(tcn.body_rotation).as_euler(
        "xyz", degrees=True
    )
    rpy_error = (estimated_rpy - gt_rpy + 180.0) % 360.0 - 180.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(run.time, run.position[:, 2], "k", lw=1.8, label="GT z")
    axes[0].plot(
        run.time,
        shallow.position[:, 2],
        color="tab:orange",
        label="shallow learned velocity z",
    )
    axes[0].plot(
        run.time,
        tcn.position[:, 2],
        color="tab:blue",
        label="Small TCN z",
    )
    for axis_index, name in enumerate(("roll", "pitch", "yaw")):
        axes[1].plot(run.time, rpy_error[:, axis_index], label=name)
    axes[0].set_ylabel("z [m]")
    axes[1].set_ylabel("RPY error [deg]")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "04_altitude_and_rpy.png", dpi=190)
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
        raise ValueError("Held-out test run cannot be used for training or bias fit.")

    runs = {
        run_id: synchronize_to_imu(load_cf231_run(dataset, run_id))
        for run_id in training_ids
    }
    sets = {
        run_id: make_sequences(
            run,
            args.window_samples,
            args.downsample,
            args.training_stride,
        )
        for run_id, run in runs.items()
    }
    cv, prediction_bias, residual_mse, selected_epochs = cross_validate(
        runs,
        sets,
        args,
    )
    final_epochs = args.final_epochs or selected_epochs
    model, stats, history = fit_model(
        [sets[run_id] for run_id in training_ids],
        args.channels,
        final_epochs,
        args.batch_size,
        args.seed,
    )

    test = synchronize_to_imu(load_cf231_run(dataset, args.test_run))
    inference = inference_sequences(test, args.window_samples, args.downsample)
    prediction = predict(model, stats, inference.windows, args.batch_size)
    training_target = np.vstack([sets[run_id].target for run_id in training_ids])
    lower = np.quantile(training_target, 0.002, axis=0)
    upper = np.quantile(training_target, 0.998, axis=0)
    prediction = np.clip(prediction, lower, upper)

    base_calibration = estimate_sensor_calibration(
        dataset,
        bias_ids,
        args.static_seconds,
    )
    pure_result, fixed_calibration = fixed_bias_attitude(
        test,
        [runs[run_id] for run_id in bias_ids],
        base_calibration,
        args.static_seconds,
    )
    tcn, world_velocity = integrate_learned_velocity(
        test,
        pure_result.trajectory,
        inference.end_indices,
        prediction,
    )
    pure, shallow = load_baselines()
    pure_metrics, _, _ = calculate_metrics(pure, test)
    shallow_metrics, _, _ = calculate_metrics(shallow, test)
    tcn_metrics, _, _ = calculate_metrics(tcn, test)
    plot_results(output, test, shallow, tcn)

    parameter_count = int(sum(value.numel() for value in model.parameters()))
    summary = {
        "configuration": {
            "dataset": str(dataset),
            "training_runs": training_ids,
            "bias_runs": bias_ids,
            "test_run": args.test_run,
            "window_samples": args.window_samples,
            "downsample": args.downsample,
            "training_stride": args.training_stride,
            "model": "SmallTCN: stem + 3 residual temporal blocks",
            "channels": args.channels,
            "parameter_count": parameter_count,
            "cv_epochs": args.cv_epochs,
            "final_epochs": final_epochs,
            "run5_gt_used_for_training": False,
            "run5_gt_used_during_inference": "initial state only; loaded afterward for scoring",
        },
        "training_samples": {
            str(run_id): int(sets[run_id].windows.shape[0]) for run_id in training_ids
        },
        "cross_validation": {
            "per_run": cv,
            "pooled_prediction_bias_m_s": prediction_bias.tolist(),
            "pooled_velocity_rmse_m_s": np.sqrt(residual_mse).tolist(),
            "pooled_bias_applied_to_run5": False,
            "bias_reason": "Per-run cross-validation biases have inconsistent signs, so a pooled correction is not transferable.",
        },
        "fixed_calibration": fixed_calibration,
        "metrics": {
            "pure_imu_full_fixed": pure_metrics,
            "extratrees_velocity_fixed_attitude": shallow_metrics,
            "small_tcn_velocity_fixed_attitude": tcn_metrics,
        },
        "interpretation": {
            "comparison": "Both learned methods use the same fixed-bias InEKF attitude; only the IMU-to-velocity regressor differs.",
            "shallow_model": "ExtraTrees on hand-compressed two-second IMU features.",
            "tcn_model": "Small temporal convolutional network on the raw two-second IMU sequence.",
        },
        "final_training_loss": history[-1],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stats": [np.asarray(value) for value in stats],
            "configuration": summary["configuration"],
            "prediction_bias_m_s": prediction_bias,
        },
        output / "small_tcn_velocity.pt",
    )
    np.savez_compressed(
        output / "trajectories.npz",
        time=test.time,
        gt_position=test.position,
        gt_rotation=test.rotation.as_matrix(),
        pure_position=pure.position,
        extratrees_position=shallow.position,
        tcn_position=tcn.position,
        tcn_velocity=tcn.velocity,
        tcn_rotation=tcn.body_rotation,
        predicted_heading_velocity=prediction,
        predicted_world_velocity=world_velocity,
        update_indices=inference.end_indices,
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
