import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")

    data: dict[str, np.ndarray] = {}
    for key in reader.fieldnames or []:
        values = []
        for row in rows:
            value = row.get(key, "")
            if value is None or value == "":
                values.append(np.nan)
            else:
                values.append(float(value))
        data[key] = np.asarray(values, dtype=np.float64)
    return data


def _detect_surface_names(columns: list[str]) -> list[str]:
    prefix = "alpha_"
    names = [column[len(prefix):] for column in columns if column.startswith(prefix)]
    if not names:
        raise ValueError("Could not detect any surface names from alpha_* columns.")
    return names


def _plot_xyz(axs, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str, ylabel: str):
    axs[0].plot(t, x, label="x")
    axs[0].set_title(title)
    axs[0].set_ylabel(ylabel)
    axs[0].legend()

    axs[1].plot(t, y, label="y", color="tab:orange")
    axs[1].set_ylabel(ylabel)
    axs[1].legend()

    axs[2].plot(t, z, label="z", color="tab:green")
    axs[2].set_ylabel(ylabel)
    axs[2].set_xlabel("time [s]")
    axs[2].legend()


def _save_or_show(fig: plt.Figure, path: Path | None, show: bool):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot per-surface aerodynamic force and torque logs.")
    parser.add_argument("csv_path", type=Path, help="Path to the surface aero CSV log.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("logs/surface_aero_plots"),
        help="Directory for generated PNG plots.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively in addition to saving them.")
    parser.add_argument("--start_time", type=float, default=None, help="Optional plot window start time [s].")
    parser.add_argument("--end_time", type=float, default=None, help="Optional plot window end time [s].")
    args = parser.parse_args()

    data = _load_csv(args.csv_path)
    surface_names = _detect_surface_names(list(data.keys()))
    time = data["sim_time_s"]

    mask = np.ones_like(time, dtype=bool)
    if args.start_time is not None:
        mask &= time >= args.start_time
    if args.end_time is not None:
        mask &= time <= args.end_time
    if not np.any(mask):
        raise ValueError("Selected time window contains no samples.")

    t = time[mask]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    _plot_xyz(
        axs,
        t,
        data["force_x"][mask],
        data["force_y"][mask],
        data["force_z"][mask],
        "Total Body Force",
        "force [N]",
    )
    _save_or_show(fig, args.out_dir / "total_force.png", args.show)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    _plot_xyz(
        axs,
        t,
        data["torque_x"][mask],
        data["torque_y"][mask],
        data["torque_z"][mask],
        "Total Body Torque",
        "torque [N m]",
    )
    _save_or_show(fig, args.out_dir / "total_torque.png", args.show)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in surface_names:
        force_mag = np.linalg.norm(
            np.stack(
                [
                    data[f"surf_force_{name}_x"][mask],
                    data[f"surf_force_{name}_y"][mask],
                    data[f"surf_force_{name}_z"][mask],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        torque_mag = np.linalg.norm(
            np.stack(
                [
                    data[f"surf_total_torque_{name}_x"][mask],
                    data[f"surf_total_torque_{name}_y"][mask],
                    data[f"surf_total_torque_{name}_z"][mask],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        axs[0].plot(t, force_mag, label=name)
        axs[1].plot(t, torque_mag, label=name)
    axs[0].set_title("Per-Surface Force Magnitude")
    axs[0].set_ylabel("force [N]")
    axs[0].legend()
    axs[1].set_title("Per-Surface Torque Magnitude")
    axs[1].set_ylabel("torque [N m]")
    axs[1].set_xlabel("time [s]")
    axs[1].legend()
    _save_or_show(fig, args.out_dir / "surface_magnitudes.png", args.show)

    for name in surface_names:
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        _plot_xyz(
            axs,
            t,
            data[f"surf_force_{name}_x"][mask],
            data[f"surf_force_{name}_y"][mask],
            data[f"surf_force_{name}_z"][mask],
            f"{name}: Surface Force",
            "force [N]",
        )
        _save_or_show(fig, args.out_dir / f"{name}_force.png", args.show)

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        _plot_xyz(
            axs,
            t,
            data[f"surf_total_torque_{name}_x"][mask],
            data[f"surf_total_torque_{name}_y"][mask],
            data[f"surf_total_torque_{name}_z"][mask],
            f"{name}: Total Torque Contribution",
            "torque [N m]",
        )
        _save_or_show(fig, args.out_dir / f"{name}_total_torque.png", args.show)

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        _plot_xyz(
            axs,
            t,
            data[f"surf_aero_torque_{name}_x"][mask],
            data[f"surf_aero_torque_{name}_y"][mask],
            data[f"surf_aero_torque_{name}_z"][mask],
            f"{name}: Aerodynamic Torque Only",
            "torque [N m]",
        )
        _save_or_show(fig, args.out_dir / f"{name}_aero_torque.png", args.show)

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        _plot_xyz(
            axs,
            t,
            data[f"surf_force_torque_{name}_x"][mask],
            data[f"surf_force_torque_{name}_y"][mask],
            data[f"surf_force_torque_{name}_z"][mask],
            f"{name}: Moment Arm Torque Only",
            "torque [N m]",
        )
        _save_or_show(fig, args.out_dir / f"{name}_force_torque.png", args.show)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
