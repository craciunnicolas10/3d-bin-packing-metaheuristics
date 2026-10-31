"""Plots: 3D packings, convergence curves and the GA vs. tabu search comparison."""
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .models import Container, Placement

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
METHOD_COLORS = {"ga": "#2a78d6", "ts": "#eb6834"}
METHOD_NAMES = {"baseline": "Greedy baseline", "ga": "Genetic algorithm", "ts": "Tabu search"}
BOX_COLORS = ["#2a78d6", "#5b9be3", "#1f5fae", "#8db8ec", "#3f87dc", "#174a88"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _box_faces(x, y, z, w, h, d):
    """Faces of a box; the container's height (y) is drawn on the vertical axis."""
    corners = [
        (x, z, y), (x + w, z, y), (x + w, z + d, y), (x, z + d, y),
        (x, z, y + h), (x + w, z, y + h), (x + w, z + d, y + h), (x, z + d, y + h),
    ]
    return [[corners[i] for i in face] for face in
            ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5))]


def _draw_packing(ax, placements: List[Placement], container: Container, title: str) -> None:
    ax.add_collection3d(Poly3DCollection(
        _box_faces(0, 0, 0, container.W, container.H, container.D),
        facecolor=(0, 0, 0, 0), edgecolor=TEXT_SECONDARY, linewidth=0.6, linestyle="--",
    ))
    for i, p in enumerate(placements):
        ax.add_collection3d(Poly3DCollection(
            _box_faces(p.x, p.y, p.z, p.w, p.h, p.d),
            facecolor=BOX_COLORS[i % len(BOX_COLORS)], edgecolor=SURFACE, linewidth=0.8, alpha=0.9,
        ))
    ax.set_xlim(0, container.W)
    ax.set_ylim(0, container.D)
    ax.set_zlim(0, container.H)
    ax.set_box_aspect((container.W, container.D, container.H))
    ax.set_xticks(np.linspace(0, container.W, 5))
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Width", labelpad=6)
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, fontsize=10, pad=-4)


def plot_packings(dataset: str, solutions: Dict[str, List[Placement]], n_boxes: int,
                  container: Container, save_path: str, show: bool = False) -> None:
    """Side-by-side 3D view of the best packing found by each method on one dataset."""
    fig = plt.figure(figsize=(9, 3.3 * len(solutions)))
    for i, (method, placements) in enumerate(solutions.items(), start=1):
        ax = fig.add_subplot(len(solutions), 1, i, projection="3d")
        utilization = 100 * sum(p.volume for p in placements) / container.volume
        _draw_packing(ax, placements, container,
                      f"{METHOD_NAMES[method]}\n{utilization:.1f}% utilization · {len(placements)}/{n_boxes} boxes")
    fig.suptitle(f"Best packing per method — {dataset} dataset\n"
                 f"container {container.W:g} × {container.D:g} × {container.H:g} (width × depth × height)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.02, hspace=0.05)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence(results: Dict[str, dict], save_path: str, show: bool = False) -> None:
    """Best-so-far utilization vs. fitness evaluations: mean over runs, with a ±1 std band."""
    names = list(results)
    cols = min(2, len(names))
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4 * rows), squeeze=False)

    for ax, name in zip(axes.flat, names):
        res = results[name]
        final = {}
        for method in ("ga", "ts"):
            runs = res[method]["runs"]
            length = min(len(r["history"]) for r in runs)
            evaluations = np.array([e for e, _ in runs[0]["history"][:length]])
            curves = np.array([[100 * f for _, f in r["history"][:length]] for r in runs])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            color = METHOD_COLORS[method]
            ax.fill_between(evaluations, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            ax.plot(evaluations, mean, color=color, linewidth=2, label=METHOD_NAMES[method])
            final[method] = (evaluations[-1], mean[-1])
        if abs(final["ga"][1] - final["ts"][1]) < 1.0:
            final = {"both": final["ga"]}
        for x_end, y_end in final.values():
            ax.annotate(f"{y_end:.1f}%", (x_end, y_end), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT_SECONDARY)
        ax.axhline(res["baseline"]["utilization"], color=TEXT_SECONDARY, linewidth=1.2,
                   linestyle="--", label=METHOD_NAMES["baseline"])
        ax.set_title(f"{name} ({res['n_boxes']} boxes)", loc="left", fontsize=11)
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Space utilization (%)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.margins(x=0.08)

    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Convergence (mean ± 1 std over runs)", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(results: Dict[str, dict], save_path: str, show: bool = False) -> None:
    """Mean utilization (with std error bars and the baseline) and mean running time per dataset."""
    names = list(results)
    x = np.arange(len(names))
    width = 0.36
    fig, (ax_util, ax_time) = plt.subplots(1, 2, figsize=(13, 4.6))

    for offset, method in ((-width / 2, "ga"), (width / 2, "ts")):
        util = [np.mean([r["utilization"] for r in results[n][method]["runs"]]) for n in names]
        util_std = [np.std([r["utilization"] for r in results[n][method]["runs"]]) for n in names]
        times = [np.mean([r["time"] for r in results[n][method]["runs"]]) for n in names]
        color = METHOD_COLORS[method]

        bars = ax_util.bar(x + offset, util, width, yerr=util_std, color=color, label=METHOD_NAMES[method],
                           error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 3})
        ax_util.bar_label(bars, labels=[f"{u:.1f}%" for u in util], padding=10, fontsize=8, color=TEXT_SECONDARY)
        bars = ax_time.bar(x + offset, times, width, color=color, label=METHOD_NAMES[method])
        ax_time.bar_label(bars, labels=[f"{t:.1f}s" for t in times], padding=2, fontsize=8, color=TEXT_SECONDARY)

    baseline = [results[n]["baseline"]["utilization"] for n in names]
    ax_util.scatter(x, baseline, marker="_", s=900, linewidths=2, color=TEXT, zorder=3, label=METHOD_NAMES["baseline"])

    ax_util.set_title("Space utilization (mean ± std)", loc="left", fontsize=11)
    ax_util.set_ylabel("Space utilization (%)")
    ax_util.set_ylim(0, 100)
    ax_time.set_title("Running time per run", loc="left", fontsize=11)
    ax_time.set_ylabel("Seconds")
    ax_time.set_ylim(0, ax_time.get_ylim()[1] * 1.15)
    for ax in (ax_util, ax_time):
        ax.set_xticks(x, names)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    handles, labels = ax_util.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Genetic algorithm vs. tabu search", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
