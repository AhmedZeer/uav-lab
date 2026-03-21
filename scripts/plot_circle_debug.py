#!/usr/bin/env python3
"""Plotter + quick analysis for spawn_uav_circle telemetry logs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_numeric_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV: {path}")
        data: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (TypeError, ValueError):
                    data[k].append(float("nan"))
    return data


def arr(data: dict[str, list[float]], key: str) -> np.ndarray:
    return np.asarray(data.get(key, []), dtype=float)


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def sign_changes(x: np.ndarray) -> int:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0
    s = np.sign(x)
    s[s == 0.0] = 1.0
    return int(np.sum(np.diff(s) != 0))


def dx_sign_flips(x: np.ndarray) -> int:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0
    dx = np.diff(x)
    s = np.sign(dx)
    s[s == 0.0] = 1.0
    return int(np.sum(np.diff(s) != 0))


def stat_line(name: str, x: np.ndarray, t: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"{name:12s} no valid data"
    duration = max(float(t[-1] - t[0]), 1.0e-9) if t.size > 1 else 1.0e-9
    zc = sign_changes(x)
    dflip = dx_sign_flips(x)
    return (
        f"{name:12s} mean={np.mean(x):9.3f} std={np.std(x):9.3f} min={np.min(x):9.3f} max={np.max(x):9.3f} "
        f"sign_changes={zc:4d} ({zc / duration:6.2f}/s) dsign_flips={dflip:4d} ({dflip / duration:6.2f}/s)"
    )


def maybe_plot(ax, t: np.ndarray, y: np.ndarray, label: str):
    tx, yy = finite_pair(t, y)
    if tx.size > 0:
        ax.plot(tx, yy, linewidth=1.0, label=label)


def resolve_default_csv() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "logs" / "circle_debug.csv",
        script_dir.parent / "logs" / "circle_debug.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main():
    default_csv = resolve_default_csv()
    parser = argparse.ArgumentParser(description="Plot and analyze circle debug telemetry CSV.")
    parser.add_argument("--csv", type=Path, default=default_csv, help=f"CSV path (default: {default_csv}).")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path (e.g. circle_plot.png).")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window.")
    parser.add_argument("--radius", type=float, default=80.0, help="Reference circle radius for XY plot [m].")
    parser.add_argument("--center_x", type=float, default=0.0, help="Reference circle center X [m].")
    parser.add_argument("--center_y", type=float, default=0.0, help="Reference circle center Y [m].")
    parser.add_argument("--t_min", type=float, default=None, help="Optional lower time bound for plotting.")
    parser.add_argument("--t_max", type=float, default=None, help="Optional upper time bound for plotting.")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = read_numeric_csv(csv_path)
    t = arr(data, "sim_time_s")
    if t.size == 0:
        raise ValueError("CSV must contain non-empty sim_time_s.")

    if args.t_min is not None or args.t_max is not None:
        t_min = float(args.t_min) if args.t_min is not None else float(t[0])
        t_max = float(args.t_max) if args.t_max is not None else float(t[-1])
        mask = (t >= t_min) & (t <= t_max)
        for k in data.keys():
            data[k] = list(np.asarray(data[k], dtype=float)[mask])
        t = arr(data, "sim_time_s")
        if t.size == 0:
            raise ValueError("No samples in requested [t_min, t_max] range.")

    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dt_median = float(np.median(np.diff(t))) if t.size > 2 else float("nan")
    print(f"CSV: {csv_path}")
    print(f"Rows: {t.size}, duration: {duration:.3f}s, median dt: {dt_median:.6f}s")
    print(stat_line("u_mps", arr(data, "u_mps"), t))
    print(stat_line("v_mps", arr(data, "v_mps"), t))
    print(stat_line("w_mps", arr(data, "w_mps"), t))
    print(stat_line("p_rps", arr(data, "p_rps"), t))
    print(stat_line("q_rps", arr(data, "q_rps"), t))
    print(stat_line("r_rps", arr(data, "r_rps"), t))
    print(stat_line("force_x", arr(data, "force_x"), t))
    print(stat_line("force_y", arr(data, "force_y"), t))
    print(stat_line("force_z", arr(data, "force_z"), t))
    print(stat_line("torque_x", arr(data, "torque_x"), t))
    print(stat_line("torque_y", arr(data, "torque_y"), t))
    print(stat_line("torque_z", arr(data, "torque_z"), t))
    print(stat_line("radius_err_m", arr(data, "radius_err_m"), t))

    fig, axes = plt.subplots(7, 1, figsize=(12, 18), constrained_layout=True)
    fig.suptitle(f"Circle Telemetry: {csv_path.name}")

    xw = arr(data, "x_w")
    yw = arr(data, "y_w")
    xw_f, yw_f = finite_pair(xw, yw)
    if xw_f.size > 0:
        axes[0].plot(xw_f, yw_f, label="trajectory", linewidth=1.1)
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        cx = float(args.center_x)
        cy = float(args.center_y)
        r = float(args.radius)
        axes[0].plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "--", label="target circle", linewidth=1.0)
    axes[0].set_ylabel("Y [m]")
    axes[0].set_xlabel("X [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    maybe_plot(axes[1], t, arr(data, "thrust_cmd"), "thrust_cmd")
    maybe_plot(axes[1], t, arr(data, "roll_cmd"), "roll_cmd")
    maybe_plot(axes[1], t, arr(data, "pitch_cmd"), "pitch_cmd")
    maybe_plot(axes[1], t, arr(data, "yaw_cmd"), "yaw_cmd")
    axes[1].set_ylabel("Commands")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    maybe_plot(axes[2], t, arr(data, "speed_mps"), "speed")
    maybe_plot(axes[2], t, arr(data, "u_mps"), "u")
    maybe_plot(axes[2], t, arr(data, "v_mps"), "v")
    maybe_plot(axes[2], t, arr(data, "w_mps"), "w")
    axes[2].set_ylabel("Velocity [m/s]")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    maybe_plot(axes[3], t, arr(data, "p_rps"), "p")
    maybe_plot(axes[3], t, arr(data, "q_rps"), "q")
    maybe_plot(axes[3], t, arr(data, "r_rps"), "r")
    axes[3].set_ylabel("Body rates [rad/s]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper right")

    maybe_plot(axes[4], t, arr(data, "force_x"), "Fx")
    maybe_plot(axes[4], t, arr(data, "force_y"), "Fy")
    maybe_plot(axes[4], t, arr(data, "force_z"), "Fz")
    axes[4].set_ylabel("Force [N]")
    axes[4].grid(alpha=0.3)
    axes[4].legend(loc="upper right")

    maybe_plot(axes[5], t, arr(data, "torque_x"), "Mx")
    maybe_plot(axes[5], t, arr(data, "torque_y"), "My")
    maybe_plot(axes[5], t, arr(data, "torque_z"), "Mz")
    axes[5].set_ylabel("Torque [N*m]")
    axes[5].grid(alpha=0.3)
    axes[5].legend(loc="upper right")

    maybe_plot(axes[6], t, arr(data, "radius_err_m"), "radius_err")
    maybe_plot(axes[6], t, arr(data, "heading_err_rad"), "heading_err")
    axes[6].set_ylabel("Circle error")
    axes[6].set_xlabel("Time [s]")
    axes[6].grid(alpha=0.3)
    axes[6].legend(loc="upper right")

    if args.save is not None:
        save_path = args.save.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
