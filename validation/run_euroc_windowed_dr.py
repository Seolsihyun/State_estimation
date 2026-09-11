"""Windowed IMU-only dead reckoning on a EuRoC sequence.

The InEKF equations are not changed. Each independent window is initialized
from GT, then only the nominal propagation used by the analytic InEKF is run.
Because there is no measurement update, covariance propagation cannot alter
the nominal trajectory and is intentionally skipped for experiment speed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters.Hoon_invariant_kalman_analytic_15D import HoonInvariantKalmanAnalytic15D
from models.Hoon_lie_group_utils import exp_so3, log_so3
from utils.imu_bias import ImuBiasEstimate, estimate_imu_bias_from_gt
from validation.euroc_loader import EurocData, load_euroc_sequence

GRAVITY = np.array([0.0, 0.0, -9.81])
BIAS_MODES = ("zero", "fixed_gt", "estimated_gt")
COLORS = {
    "zero": "tab:gray",
    "fixed_gt": "tab:blue",
    "estimated_gt": "tab:orange",
}


@dataclass(frozen=True)
class BiasSet:
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class ConsistencyResult:
    time: np.ndarray
    gyro_residual: np.ndarray
    accel_residual: np.ndarray
    gt_gyro_bias: np.ndarray
    gt_accel_bias: np.ndarray


def make_engine() -> HoonInvariantKalmanAnalytic15D:
    return HoonInvariantKalmanAnalytic15D(
        mode="imu_only",
        motion_config={
            "gravity": GRAVITY,
            "jacobian_mode": "analytic",
            "translation_input_frame": "body",
            "translation_input_type": "acceleration",
            "rotation_input_type": "rate",
        },
    )


def imu_gt_consistency(data: EurocData) -> ConsistencyResult:
    """Compare interval-centered IMU samples with GT-implied measurements."""
    dt = np.diff(data.time)
    if np.any(dt <= 0.0):
        raise ValueError("Non-positive synchronized timestamp interval.")
    imu_mid = 0.5 * (data.imu[:-1] + data.imu[1:])
    velocity_dot = np.diff(data.velocity, axis=0) / dt[:, None]
    rotations = data.rotation.as_matrix()
    rotation_mid = np.empty((len(dt), 3, 3), dtype=float)
    omega_gt = np.empty((len(dt), 3), dtype=float)
    for index in range(len(dt)):
        rotvec = log_so3(rotations[index].T @ rotations[index + 1])
        omega_gt[index] = rotvec / dt[index]
        rotation_mid[index] = rotations[index] @ exp_so3(0.5 * rotvec)
    specific_force_gt = np.einsum(
        "nji,nj->ni",
        rotation_mid,
        velocity_dot - GRAVITY[None, :],
    )
    return ConsistencyResult(
        time=0.5 * (data.time[:-1] + data.time[1:]),
        gyro_residual=imu_mid[:, 3:6] - omega_gt,
        accel_residual=imu_mid[:, 0:3] - specific_force_gt,
        gt_gyro_bias=0.5 * (data.gyro_bias[:-1] + data.gyro_bias[1:]),
        gt_accel_bias=0.5 * (data.accel_bias[:-1] + data.accel_bias[1:]),
    )


def calibration_biases(
    data: EurocData,
    calibration_seconds: float,
) -> tuple[dict[str, BiasSet], ImuBiasEstimate]:
    calibration_end = min(float(calibration_seconds), float(data.time[-1]))
    mask = data.time <= calibration_end
    if np.count_nonzero(mask) < 2:
        raise ValueError("Calibration interval contains fewer than two samples.")
    fixed_gt = BiasSet(
        gyro=np.mean(data.gyro_bias[mask], axis=0),
        accel=np.mean(data.accel_bias[mask], axis=0),
    )
    estimated = estimate_imu_bias_from_gt(
        data.time,
        data.imu,
        data.rotation.as_matrix(),
        data.velocity,
        gravity=GRAVITY,
        calibration_range=(float(data.time[0]), calibration_end),
        statistic="median",
    )
    return (
        {
            "zero": BiasSet(np.zeros(3), np.zeros(3)),
            "fixed_gt": fixed_gt,
            "estimated_gt": BiasSet(estimated.gyro, estimated.accel),
        },
        estimated,
    )


def select_bias(
    mode: str,
    start_index: int,
    data: EurocData,
    fixed_biases: dict[str, BiasSet],
) -> BiasSet:
    del start_index, data
    return fixed_biases[mode]


def propagate_window(
    engine: HoonInvariantKalmanAnalytic15D,
    data: EurocData,
    start: int,
    end: int,
    bias: BiasSet,
    *,
    save_trajectory: bool = False,
    imu_sampling: str = "endpoint",
) -> dict[str, np.ndarray]:
    """Propagate one GT-initialized window without measurements."""
    if imu_sampling not in {"previous", "endpoint", "midpoint"}:
        raise ValueError(
            "imu_sampling must be 'previous', 'endpoint', or 'midpoint'."
        )
    rotation = data.rotation[start].as_matrix().copy()
    velocity = data.velocity[start].copy()
    position = data.position[start].copy()
    rotations = [rotation.copy()] if save_trajectory else None
    velocities = [velocity.copy()] if save_trajectory else None
    positions = [position.copy()] if save_trajectory else None

    for index in range(start + 1, end + 1):
        dt = float(data.time[index] - data.time[index - 1])
        control = data.imu[index - 1]
        if imu_sampling == "endpoint":
            control = data.imu[index]
        elif imu_sampling == "midpoint":
            # The propagation assumes one constant input over [t[k-1], t[k]].
            # Averaging the two timestamped samples places that input at the
            # interval center without modifying the InEKF state equations.
            control = 0.5 * (data.imu[index - 1] + data.imu[index])
        rotation, velocity, position = engine._propagate_nominal(
            rotation,
            velocity,
            position,
            bias.gyro,
            bias.accel,
            control,
            dt,
        )
        if save_trajectory:
            rotations.append(rotation.copy())
            velocities.append(velocity.copy())
            positions.append(position.copy())

    output = {
        "rotation": rotation,
        "velocity": velocity,
        "position": position,
    }
    if save_trajectory:
        output.update(
            rotations=np.asarray(rotations),
            velocities=np.asarray(velocities),
            positions=np.asarray(positions),
            time=data.time[start : end + 1] - data.time[start],
        )
    return output


def window_error(
    data: EurocData,
    start: int,
    end: int,
    propagated: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    position_error = float(np.linalg.norm(propagated["position"] - data.position[end]))
    velocity_error = float(np.linalg.norm(propagated["velocity"] - data.velocity[end]))
    attitude_error = float(
        np.rad2deg(
            np.linalg.norm(
                log_so3(data.rotation[end].as_matrix().T @ propagated["rotation"])
            )
        )
    )
    return position_error, velocity_error, attitude_error


def generate_window_indices(
    data: EurocData,
    duration: float,
    stride: float,
    evaluation_start: float,
    evaluation_end: float,
    max_windows: int,
) -> list[tuple[int, int]]:
    last_start = evaluation_end - duration
    if last_start < evaluation_start:
        return []
    requested_starts = np.arange(evaluation_start, last_start + 1e-9, stride)
    windows: list[tuple[int, int]] = []
    previous_start = -1
    for requested_start in requested_starts:
        start = int(np.searchsorted(data.time, requested_start, side="left"))
        end = int(np.searchsorted(data.time, data.time[start] + duration, side="left"))
        if end >= len(data.time) or data.time[end] > evaluation_end + 1e-9:
            continue
        if start == previous_start:
            continue
        windows.append((start, end))
        previous_start = start
        if max_windows > 0 and len(windows) >= max_windows:
            break
    return windows


def run_window_experiments(
    data: EurocData,
    fixed_biases: dict[str, BiasSet],
    durations: list[float],
    stride: float,
    evaluation_start: float,
    evaluation_end: float,
    moving_threshold: float,
    max_gap: float,
    max_windows: int,
    imu_sampling: str,
) -> list[dict[str, float | str | bool]]:
    engine = make_engine()
    rows: list[dict[str, float | str | bool]] = []
    for duration in durations:
        windows = generate_window_indices(
            data,
            duration,
            stride,
            evaluation_start,
            evaluation_end,
            max_windows,
        )
        print(f"  {duration:g}s: {len(windows)} windows x {len(BIAS_MODES)} bias modes")
        for start, end in windows:
            interval_dt = np.diff(data.time[start : end + 1])
            mean_speed = float(
                np.mean(np.linalg.norm(data.velocity[start : end + 1], axis=1))
            )
            base = {
                "duration": float(duration),
                "start_time": float(data.time[start]),
                "end_time": float(data.time[end]),
                "start_index": int(start),
                "end_index": int(end),
                "mean_speed": mean_speed,
                "moving": bool(mean_speed >= moving_threshold),
                "max_dt": float(np.max(interval_dt)),
                "has_large_gap": bool(np.max(interval_dt) > max_gap),
            }
            for mode in BIAS_MODES:
                bias = select_bias(mode, start, data, fixed_biases)
                propagated = propagate_window(
                    engine,
                    data,
                    start,
                    end,
                    bias,
                    imu_sampling=imu_sampling,
                )
                ep, ev, er = window_error(data, start, end, propagated)
                rows.append(
                    {
                        **base,
                        "bias_mode": mode,
                        "position_error": ep,
                        "velocity_error": ev,
                        "attitude_error_deg": er,
                    }
                )
    return rows


def summarize_rows(rows: list[dict], moving_only: bool) -> list[dict]:
    summaries: list[dict] = []
    durations = sorted({float(row["duration"]) for row in rows})
    for duration in durations:
        for mode in BIAS_MODES:
            selected = [
                row
                for row in rows
                if float(row["duration"]) == duration
                and row["bias_mode"] == mode
                and (not moving_only or bool(row["moving"]))
                and not bool(row["has_large_gap"])
            ]
            if not selected:
                continue
            summary: dict[str, float | str | int | bool] = {
                "duration": duration,
                "bias_mode": mode,
                "moving_only": moving_only,
                "count": len(selected),
            }
            for metric_name in [
                "position_error",
                "velocity_error",
                "attitude_error_deg",
            ]:
                values = np.asarray([float(row[metric_name]) for row in selected])
                summary.update(
                    {
                        f"{metric_name}_median": float(np.median(values)),
                        f"{metric_name}_q25": float(np.percentile(values, 25)),
                        f"{metric_name}_q75": float(np.percentile(values, 75)),
                        f"{metric_name}_p95": float(np.percentile(values, 95)),
                        f"{metric_name}_max": float(np.max(values)),
                    }
                )
            summaries.append(summary)
    return summaries


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_consistency(
    result: ConsistencyResult,
    calibration_end: float,
    output: Path,
) -> dict[str, list[float] | float]:
    calibration = result.time <= calibration_end
    gyro_delta = result.gyro_residual - result.gt_gyro_bias
    accel_delta = result.accel_residual - result.gt_accel_bias
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    for axis, label in enumerate(labels):
        axes[axis, 0].plot(result.time, result.gyro_residual[:, axis], lw=.7, label="IMU-GT")
        axes[axis, 0].plot(result.time, result.gt_gyro_bias[:, axis], lw=1, label="GT bias")
        axes[axis, 0].axvspan(0, calibration_end, color="tab:green", alpha=.12)
        axes[axis, 0].set_ylabel(f"gyro {label}\n[rad/s]"); axes[axis, 0].grid(True)
        axes[axis, 1].plot(result.time, result.accel_residual[:, axis], lw=.7, label="IMU-GT")
        axes[axis, 1].plot(result.time, result.gt_accel_bias[:, axis], lw=1, label="GT bias")
        axes[axis, 1].axvspan(0, calibration_end, color="tab:green", alpha=.12)
        axes[axis, 1].set_ylabel(f"accel {label}\n[m/s²]"); axes[axis, 1].grid(True)
    axes[0, 0].set_title("Gyroscope residual consistency")
    axes[0, 1].set_title("Accelerometer residual consistency")
    axes[0, 0].legend(); axes[0, 1].legend()
    axes[-1, 0].set_xlabel("time [s]"); axes[-1, 1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
    return {
        "calibration_gyro_residual_minus_bias_median": np.median(
            gyro_delta[calibration], axis=0
        ).tolist(),
        "calibration_accel_residual_minus_bias_median": np.median(
            accel_delta[calibration], axis=0
        ).tolist(),
        "full_gyro_residual_minus_bias_rmse": float(np.sqrt(np.mean(gyro_delta**2))),
        "full_accel_residual_minus_bias_rmse": float(np.sqrt(np.mean(accel_delta**2))),
    }


def grouped_boxplot(
    rows: list[dict],
    metric: str,
    ylabel: str,
    output: Path,
    moving_only: bool,
) -> None:
    durations = sorted({float(row["duration"]) for row in rows})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.17
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(BIAS_MODES))
    for mode_index, mode in enumerate(BIAS_MODES):
        values_by_duration = []
        positions = []
        for duration_index, duration in enumerate(durations):
            selected = [
                float(row[metric])
                for row in rows
                if float(row["duration"]) == duration
                and row["bias_mode"] == mode
                and not bool(row["has_large_gap"])
                and (not moving_only or bool(row["moving"]))
            ]
            if selected:
                values_by_duration.append(selected)
                positions.append(duration_index + offsets[mode_index])
        if values_by_duration:
            plot = ax.boxplot(
                values_by_duration,
                positions=positions,
                widths=width * .9,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for box in plot["boxes"]:
                box.set_facecolor(COLORS[mode]); box.set_alpha(.75)
            ax.plot([], [], color=COLORS[mode], lw=8, alpha=.75, label=mode)
    ax.set_xticks(range(len(durations)), [f"{value:g}" for value in durations])
    ax.set_xlabel("window duration [s]"); ax.set_ylabel(ylabel)
    ax.set_title(("Moving-window " if moving_only else "All-window ") + ylabel)
    ax.grid(True, axis="y"); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def representative_rows(rows: list[dict], duration: float) -> dict[str, dict]:
    candidates = [
        row
        for row in rows
        if float(row["duration"]) == duration
        and bool(row["moving"])
        and not bool(row["has_large_gap"])
    ]
    chosen: dict[str, dict] = {}
    for mode in BIAS_MODES:
        mode_rows = [row for row in candidates if row["bias_mode"] == mode]
        if not mode_rows:
            continue
        median = float(np.median([float(row["position_error"]) for row in mode_rows]))
        chosen[mode] = min(
            mode_rows,
            key=lambda row: abs(float(row["position_error"]) - median),
        )
    return chosen


def plot_representative_trajectories(
    data: EurocData,
    rows: list[dict],
    fixed_biases: dict[str, BiasSet],
    durations: list[float],
    output: Path,
    imu_sampling: str,
    *,
    corrected_only: bool = False,
) -> None:
    engine = make_engine()
    fig, axes = plt.subplots(1, len(durations), figsize=(6 * len(durations), 5))
    axes = np.atleast_1d(axes)
    for ax, duration in zip(axes, durations):
        chosen = representative_rows(rows, duration)
        if not chosen:
            ax.set_title(f"{duration:g}s: no moving window"); ax.axis("off"); continue
        # Use the fixed-GT median window as the common representative start.
        anchor = chosen.get("fixed_gt", next(iter(chosen.values())))
        start, end = int(anchor["start_index"]), int(anchor["end_index"])
        R0 = data.rotation[start].as_matrix()
        gt_relative = (R0.T @ (data.position[start : end + 1] - data.position[start]).T).T
        ax.plot(gt_relative[:, 0], gt_relative[:, 1], "k--", lw=2, label="GT")
        plotted_modes = BIAS_MODES[1:] if corrected_only else BIAS_MODES
        for mode in plotted_modes:
            bias = select_bias(mode, start, data, fixed_biases)
            trajectory = propagate_window(
                engine,
                data,
                start,
                end,
                bias,
                save_trajectory=True,
                imu_sampling=imu_sampling,
            )
            relative = (
                R0.T
                @ (trajectory["positions"] - trajectory["positions"][0]).T
            ).T
            ax.plot(relative[:, 0], relative[:, 1], color=COLORS[mode], label=mode)
        ax.scatter([0], [0], color="black", s=25)
        suffix = " (bias-corrected zoom)" if corrected_only else ""
        ax.set_title(f"Representative {duration:g}s moving window{suffix}")
        ax.set_xlabel("initial-body x [m]"); ax.set_ylabel("initial-body y [m]")
        ax.axis("equal"); ax.grid(True)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_one_second_global_segments(
    data: EurocData,
    rows: list[dict],
    fixed_biases: dict[str, BiasSet],
    output: Path,
    imu_sampling: str,
) -> None:
    """Show all independent one-second estimated-GT-bias DR segments."""
    engine = make_engine()
    candidates = [
        row
        for row in rows
        if float(row["duration"]) == 1.0
        and row["bias_mode"] == "estimated_gt"
        and bool(row["moving"])
        and not bool(row["has_large_gap"])
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(data.position[:, 0], data.position[:, 1], "k--", lw=1.5, label="GT")
    bias = fixed_biases["estimated_gt"]
    for segment_index, row in enumerate(candidates):
        trajectory = propagate_window(
            engine,
            data,
            int(row["start_index"]),
            int(row["end_index"]),
            bias,
            save_trajectory=True,
            imu_sampling=imu_sampling,
        )
        ax.plot(
            trajectory["positions"][:, 0],
            trajectory["positions"][:, 1],
            color="tab:orange",
            lw=1.0,
            alpha=0.8,
            label="1 s IMU-only segment" if segment_index == 0 else None,
        )
    ax.set_title("EuRoC V1_01: independent 1 s IMU-only dead-reckoning segments")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_biases(
    data: EurocData,
    fixed_biases: dict[str, BiasSet],
    estimated: ImuBiasEstimate,
    calibration_end: float,
    output: Path,
) -> None:
    labels = ["x", "y", "z"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, gt, fixed, estimate, title, unit in [
        (
            axes[0],
            data.gyro_bias,
            fixed_biases["fixed_gt"].gyro,
            estimated.gyro,
            "Gyroscope bias",
            "rad/s",
        ),
        (
            axes[1],
            data.accel_bias,
            fixed_biases["fixed_gt"].accel,
            estimated.accel,
            "Accelerometer bias",
            "m/s²",
        ),
    ]:
        mask = data.time <= calibration_end
        ax.bar(x - .2, np.mean(gt[mask], axis=0), .4, label="GT calibration mean")
        ax.bar(x + .2, estimate, .4, label="GT-residual estimate")
        ax.set_xticks(x, labels); ax.set_title(title); ax.set_ylabel(unit); ax.grid(True, axis="y")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def write_report(
    path: Path,
    data: EurocData,
    args: argparse.Namespace,
    consistency_metrics: dict,
    fixed_biases: dict[str, BiasSet],
    estimated: ImuBiasEstimate,
    summaries: list[dict],
) -> None:
    moving = [summary for summary in summaries if summary["moving_only"]]
    lines = [
        "# EuRoC windowed IMU-only dead-reckoning",
        "",
        f"- Sequence: `{data.source_dir}`",
        f"- Samples: {len(data.time)}",
        f"- Duration: {data.time[-1]:.3f} s",
        f"- Calibration: first {args.calibration_seconds:g} s",
        f"- Windows: {', '.join(f'{value:g}s' for value in args.window_lengths)}; stride {args.stride:g} s",
        f"- IMU interval sampling: `{args.imu_sampling}`",
        "- Every window is initialized from GT. No measurement is used inside a window.",
        "- The same analytic InEKF nominal propagation is used; covariance is skipped because it cannot affect an IMU-only nominal trajectory.",
        "",
        "## Fixed calibration biases",
        "",
        f"- GT gyro mean: `{np.array2string(fixed_biases['fixed_gt'].gyro, precision=8)}`",
        f"- GT accel mean: `{np.array2string(fixed_biases['fixed_gt'].accel, precision=8)}`",
        f"- GT-residual gyro estimate: `{np.array2string(estimated.gyro, precision=8)}`",
        f"- GT-residual accel estimate: `{np.array2string(estimated.accel, precision=8)}`",
        "",
        "## IMU–GT consistency",
        "",
        f"- Full gyro residual-minus-bias RMSE: {consistency_metrics['full_gyro_residual_minus_bias_rmse']:.6g} rad/s",
        f"- Full accel residual-minus-bias RMSE: {consistency_metrics['full_accel_residual_minus_bias_rmse']:.6g} m/s²",
        "",
        "## Moving-window median errors",
        "",
        "| Window | Bias | Count | Position [m] | Velocity [m/s] | Attitude [deg] |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for summary in moving:
        lines.append(
            f"| {summary['duration']:g}s | {summary['bias_mode']} | {summary['count']} | "
            f"{summary['position_error_median']:.6g} | "
            f"{summary['velocity_error_median']:.6g} | "
            f"{summary['attitude_error_deg_median']:.6g} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `zero`: no bias correction.",
        "- `fixed_gt`: one GT-bias mean from the calibration interval, fixed for every window.",
        "- `estimated_gt`: one bias estimated from IMU–GT residuals, fixed for every window.",
        "- Large-gap windows are stored in CSV but excluded from summary plots.",
        "- Full-sequence integration is intentionally not the primary success criterion; window length quantifies the useful IMU-only horizon.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=ROOT / "data" / "euroc" / "V1_01_easy",
        help="EuRoC sequence directory or its mav0 directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "results" / "euroc_v1_01",
    )
    parser.add_argument("--window-lengths", type=float, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--calibration-seconds", type=float, default=5.0)
    parser.add_argument(
        "--evaluation-start",
        type=float,
        default=None,
        help="Defaults to the end of calibration.",
    )
    parser.add_argument("--evaluation-end", type=float, default=None)
    parser.add_argument("--moving-threshold", type=float, default=0.05)
    parser.add_argument("--max-gap", type=float, default=0.03)
    parser.add_argument(
        "--imu-sampling",
        choices=["previous", "endpoint", "midpoint"],
        default="endpoint",
        help="Constant IMU input assigned to each integration interval.",
    )
    parser.add_argument(
        "--max-windows-per-duration",
        type=int,
        default=0,
        help="0 uses all windows; useful nonzero value for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading EuRoC sequence: {args.sequence}")
    data = load_euroc_sequence(args.sequence)
    evaluation_start = (
        args.calibration_seconds
        if args.evaluation_start is None
        else args.evaluation_start
    )
    evaluation_end = (
        float(data.time[-1]) if args.evaluation_end is None else args.evaluation_end
    )
    if not 0 <= evaluation_start < evaluation_end <= data.time[-1] + 1e-9:
        raise ValueError(
            f"Invalid evaluation range [{evaluation_start}, {evaluation_end}] "
            f"for data duration {data.time[-1]}."
        )

    print("Checking IMU–GT consistency and preparing bias modes...")
    consistency = imu_gt_consistency(data)
    fixed_biases, estimated = calibration_biases(data, args.calibration_seconds)
    consistency_metrics = plot_consistency(
        consistency,
        args.calibration_seconds,
        figures / "01_imu_gt_consistency.png",
    )
    plot_biases(
        data,
        fixed_biases,
        estimated,
        args.calibration_seconds,
        figures / "02_bias_comparison.png",
    )

    print("Running windowed IMU-only propagation...")
    rows = run_window_experiments(
        data,
        fixed_biases,
        sorted(set(args.window_lengths)),
        args.stride,
        evaluation_start,
        evaluation_end,
        args.moving_threshold,
        args.max_gap,
        args.max_windows_per_duration,
        args.imu_sampling,
    )
    if not rows:
        raise RuntimeError("No valid evaluation windows were generated.")
    save_rows(args.output / "window_metrics.csv", rows)
    summaries = summarize_rows(rows, moving_only=False) + summarize_rows(
        rows, moving_only=True
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "sequence": str(data.source_dir),
                    "window_lengths": args.window_lengths,
                    "stride": args.stride,
                    "calibration_seconds": args.calibration_seconds,
                    "evaluation_start": evaluation_start,
                    "evaluation_end": evaluation_end,
                    "moving_threshold": args.moving_threshold,
                    "max_gap": args.max_gap,
                    "imu_sampling": args.imu_sampling,
                },
                "consistency": consistency_metrics,
                "fixed_biases": {
                    mode: {
                        "gyro": bias.gyro.tolist(),
                        "accel": bias.accel.tolist(),
                    }
                    for mode, bias in fixed_biases.items()
                },
                "estimated_bias": {
                    "gyro": estimated.gyro.tolist(),
                    "accel": estimated.accel.tolist(),
                    "gyro_residual_std": estimated.gyro_residual_std.tolist(),
                    "accel_residual_std": estimated.accel_residual_std.tolist(),
                },
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    grouped_boxplot(
        rows,
        "position_error",
        "final position error [m]",
        figures / "03_position_error_all.png",
        moving_only=False,
    )
    grouped_boxplot(
        rows,
        "position_error",
        "final position error [m]",
        figures / "04_position_error_moving.png",
        moving_only=True,
    )
    grouped_boxplot(
        rows,
        "attitude_error_deg",
        "final attitude error [deg]",
        figures / "05_attitude_error_moving.png",
        moving_only=True,
    )
    plot_representative_trajectories(
        data,
        rows,
        fixed_biases,
        sorted(set(args.window_lengths)),
        figures / "06_representative_trajectories.png",
        args.imu_sampling,
    )
    plot_representative_trajectories(
        data,
        rows,
        fixed_biases,
        sorted(set(args.window_lengths)),
        figures / "07_bias_corrected_trajectories.png",
        args.imu_sampling,
        corrected_only=True,
    )
    if 1.0 in args.window_lengths:
        plot_one_second_global_segments(
            data,
            rows,
            fixed_biases,
            figures / "08_one_second_global_segments.png",
            args.imu_sampling,
        )
    write_report(
        args.output / "report.md",
        data,
        args,
        consistency_metrics,
        fixed_biases,
        estimated,
        summaries,
    )
    print(f"Done. Results: {args.output}")


if __name__ == "__main__":
    main()
