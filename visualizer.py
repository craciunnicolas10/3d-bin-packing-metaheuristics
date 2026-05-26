from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D               # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models import Container, Placement


def _draw_box_3d(ax, x, y, z, w, h, d, color, alpha=0.7):
    verts = [
        [x,     y,     z],      [x + w, y,     z],
        [x + w, y + h, z],      [x,     y + h, z],
        [x,     y,     z + d],  [x + w, y,     z + d],
        [x + w, y + h, z + d],  [x,     y + h, z + d],
    ]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor="black", linewidth=0.3)
    ax.add_collection3d(poly)


def plot_solution_3d(
    placements: List[Placement],
    container: Container,
    title: str = "3D Bin Packing",
    save_path: str = None
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.get_cmap("tab20", max(len(placements), 1))

    _draw_box_3d(ax, 0, 0, 0, container.W, container.H, container.D, color="lightgray", alpha=0.05)

    for i, p in enumerate(placements):
        _draw_box_3d(ax, p.x, p.y, p.z, p.w, p.h, p.d, color=cmap(i), alpha=0.75)

    ax.set_xlim(0, container.W)
    ax.set_ylim(0, container.H)
    ax.set_zlim(0, container.D)
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("D")

    placed_vol = sum(p.w * p.h * p.d for p in placements)
    util_pct = 100.0 * placed_vol / container.volume
    ax.set_title(f"{title}\n{len(placements)} cutii | Utilizare: {util_pct:.2f}%", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Salvat: {save_path}")
    plt.show()


def plot_convergence(results: Dict[str, dict], save_path: str = "convergenta.png") -> None:
    dataset_names = list(results.keys())
    n = len(dataset_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Convergența AG vs. Tabu Search", fontsize=13, fontweight="bold")

    for ax, dname in zip(axes, dataset_names):
        res = results[dname]
        ga_h = [v * 100 for v in res["ga"]["history"]]
        ts_h = [v * 100 for v in res["ts"]["history"]]

        ax.plot(ga_h, color="#2563EB", linewidth=2, label=f"AG ({ga_h[-1]:.1f}%)")
        ax.plot(ts_h, color="#DC2626", linewidth=2, linestyle="--", label=f"TS ({ts_h[-1]:.1f}%)")
        ax.set_title(f"{dname} ({res['n_boxes']} cutii)")
        ax.set_xlabel("Iterații")
        ax.set_ylabel("Utilizare (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Salvat: {save_path}")
    plt.show()


def plot_comparison_bars(results: Dict[str, dict], save_path: str = "comparare.png") -> None:
    dataset_names = list(results.keys())
    ga_utils = [results[d]["ga"]["space_util"] for d in dataset_names]
    ts_utils = [results[d]["ts"]["space_util"] for d in dataset_names]
    ga_times = [results[d]["ga"]["time"]        for d in dataset_names]
    ts_times = [results[d]["ts"]["time"]        for d in dataset_names]

    x = np.arange(len(dataset_names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparare AG vs. Tabu Search", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x - w/2, ga_utils, w, label="AG",   color="#2563EB", alpha=0.85)
    bars2 = ax1.bar(x + w/2, ts_utils, w, label="TS",   color="#DC2626", alpha=0.85)
    ax1.set_ylabel("Utilizare (%)")
    ax1.set_title("Utilizare spațiu")
    ax1.set_xticks(x); ax1.set_xticklabels(dataset_names)
    ax1.set_ylim(0, 100); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=9)

    bars3 = ax2.bar(x - w/2, ga_times, w, label="AG",   color="#2563EB", alpha=0.85)
    bars4 = ax2.bar(x + w/2, ts_times, w, label="TS",   color="#DC2626", alpha=0.85)
    ax2.set_ylabel("Timp (s)")
    ax2.set_title("Timp execuție")
    ax2.set_xticks(x); ax2.set_xticklabels(dataset_names)
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{bar.get_height():.1f}s", ha="center", fontsize=9)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{bar.get_height():.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Salvat: {save_path}")
    plt.show()