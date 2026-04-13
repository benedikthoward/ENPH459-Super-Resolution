"""
Plot pixel shift vs tilt angle from pinhole calibration data.

Reads shifts.csv produced by calibrate_shift_grid.py and generates
a 2x3 figure: position grid key + dx/dy vs tilt for X-sweep and Y-sweep.

Usage:
    python plot_beam_shifts.py [--data-dir ./data] [--output-dir ./results]
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 9-position grid labels (same as calibrate_shift_grid.py)
GRID_SIGNS = [(sx, sy) for sy in [1, 0, -1] for sx in [-1, 0, 1]]
GRID_LABELS = [
    "(-x,+y)", "(0,+y)", "(+x,+y)",
    "(-x, 0)", "(0, 0)", "(+x, 0)",
    "(-x,-y)", "(0,-y)", "(+x,-y)",
]
CENTER_IDX = 4


def load_shifts(csv_path):
    """Load shifts.csv into {sweep_axis: {tilt: {pos: {dx_mean, dy_mean, ...}}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            axis = row["sweep_axis"]
            tilt = float(row["tilt_angle_deg"])
            pos = int(row["position"])
            data[axis][tilt][pos] = {
                "dx_mean": float(row["dx_mean_px"]),
                "dx_std": float(row["dx_std_px"]),
                "dy_mean": float(row["dy_mean_px"]),
                "dy_std": float(row["dy_std_px"]),
                "label": row["label"],
            }
    return data


def plot_results(data, out_path):
    pos_indices = [p for p in range(9) if p != CENTER_IDX]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pos_indices)))
    pos_colors = {p: colors[i] for i, p in enumerate(pos_indices)}
    pos_markers = {p: m for p, m in zip(pos_indices, ["o", "s", "^", "D", "v", "<", ">", "p"])}

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2, 2])

    # Position diagram
    ax_diagram = fig.add_subplot(gs[:, 0])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect("equal")
    ax_diagram.set_title("Position Grid Key")
    ax_diagram.set_xlabel("X sign")
    ax_diagram.set_ylabel("Y sign")

    for p, (sx, sy) in enumerate(GRID_SIGNS):
        if p == CENTER_IDX:
            ax_diagram.plot(sx, sy, marker="x", color="gray", markersize=15, markeredgewidth=3)
            ax_diagram.annotate("REF", (sx, sy), textcoords="offset points",
                                xytext=(12, -5), fontsize=9, color="gray")
        else:
            ax_diagram.plot(sx, sy, marker=pos_markers[p], color=pos_colors[p],
                            markersize=12, markeredgewidth=2)
            ax_diagram.annotate(GRID_LABELS[p], (sx, sy), textcoords="offset points",
                                xytext=(12, -5), fontsize=8, color=pos_colors[p])

    ax_diagram.grid(True, alpha=0.3)
    ax_diagram.set_xticks([-1, 0, 1])
    ax_diagram.set_yticks([-1, 0, 1])

    def _plot_sweep(ax_dx, ax_dy, results, title_prefix):
        tilt_angles = sorted(results.keys())
        for p in pos_indices:
            dx_means = [results[t][p]["dx_mean"] for t in tilt_angles if p in results[t]]
            dx_stds = [results[t][p]["dx_std"] for t in tilt_angles if p in results[t]]
            dy_means = [results[t][p]["dy_mean"] for t in tilt_angles if p in results[t]]
            dy_stds = [results[t][p]["dy_std"] for t in tilt_angles if p in results[t]]
            tilts = [t for t in tilt_angles if p in results[t]]

            ax_dx.errorbar(tilts, dx_means, yerr=dx_stds,
                           fmt="-", marker=pos_markers[p], color=pos_colors[p],
                           label=GRID_LABELS[p], markersize=4, capsize=3)
            ax_dy.errorbar(tilts, dy_means, yerr=dy_stds,
                           fmt="-", marker=pos_markers[p], color=pos_colors[p],
                           label=GRID_LABELS[p], markersize=4, capsize=3)

        for ax, comp in [(ax_dx, "dx"), (ax_dy, "dy")]:
            ax.axhline(y=0, color="k", alpha=0.2)
            ax.set_xlabel("Tilt angle (degrees)")
            ax.set_ylabel(f"{comp} shift (px)")
            ax.set_title(f"{title_prefix}: {comp} vs tilt angle")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

    # X-sweep
    ax_dx_x = fig.add_subplot(gs[0, 1])
    ax_dy_x = fig.add_subplot(gs[0, 2])
    _plot_sweep(ax_dx_x, ax_dy_x, data["x"], "X-sweep")

    # Y-sweep
    ax_dx_y = fig.add_subplot(gs[1, 1])
    ax_dy_y = fig.add_subplot(gs[1, 2])
    _plot_sweep(ax_dx_y, ax_dy_y, data["y"], "Y-sweep")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot beam shift calibration data")
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent / "data"),
                        help="Directory containing shifts.csv")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "results"),
                        help="Directory to save plot")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shifts = load_shifts(data_dir / "shifts.csv")
    plot_results(shifts, out_dir / "pinhole_shifts.png")
