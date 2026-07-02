"""Shared five-block x-axis geometry for figures spanning flat trail, stairs, and ramp.

Consumed by fig6_elevation_physiology_composite and fig7_eda_full_session.
This module contains only pure geometry helpers — no data, no matplotlib state.
"""
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

TASK_ORDER: List[str] = ["flat_trail", "stairs", "ramp"]
TASK_LABELS: Dict[str, str] = {
    "flat_trail": "Flat Trail",
    "stairs":     "Stairs",
    "ramp":       "Ramp",
}

TASK_BLOCK_WIDTH:  float = 1.0
BREAK_BLOCK_WIDTH: float = 0.18

_BLOCK_KEYS = ["flat_trail", "break_1", "stairs", "break_2", "ramp"]


def block_layout(
    task_distances_m: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return global x-axis [start, end) for each of the five blocks.

    When task_distances_m is provided, task-block widths scale proportionally
    to path length (longer path → more x-space). Without it every task block
    gets TASK_BLOCK_WIDTH, appropriate when all tasks are equal duration.
    """
    if task_distances_m:
        total = sum(task_distances_m.get(t, 1.0) for t in TASK_ORDER)
        task_ws = [
            task_distances_m.get(t, 1.0) / total * len(TASK_ORDER) * TASK_BLOCK_WIDTH
            for t in TASK_ORDER
        ]
        widths = [task_ws[0], BREAK_BLOCK_WIDTH, task_ws[1], BREAK_BLOCK_WIDTH, task_ws[2]]
    else:
        widths = [TASK_BLOCK_WIDTH, BREAK_BLOCK_WIDTH, TASK_BLOCK_WIDTH, BREAK_BLOCK_WIDTH, TASK_BLOCK_WIDTH]

    blocks: Dict[str, Tuple[float, float]] = {}
    x = 0.0
    for key, width in zip(_BLOCK_KEYS, widths):
        blocks[key] = (x, x + width)
        x += width
    return blocks


def task_x(
    task_label: str,
    position_pct: np.ndarray,
    blocks: Dict[str, Tuple[float, float]],
    padding: float = 0.0,
) -> np.ndarray:
    """Map a 0-100 % position within a task onto that block's global x-range.

    padding: x-units of whitespace to leave on each side of the block so
    plotted content does not run flush against the adjacent break-block edges.
    """
    start, end = blocks[task_label]
    return (start + padding) + (position_pct / 100.0) * (end - start - 2 * padding)


def draw_break_blocks(ax: plt.Axes, blocks: Dict[str, Tuple[float, float]]) -> None:
    """Draw cross-hatched break blocks on ax (no EDA/elevation inside)."""
    for key in ("break_1", "break_2"):
        start, end = blocks[key]
        ax.axvspan(
            start, end,
            facecolor="none", edgecolor="#999999", hatch="////", linewidth=0.5, zorder=1,
        )


def draw_block_labels(
    ax: plt.Axes,
    blocks: Dict[str, Tuple[float, float]],
    present_tasks: List[str],
    y: float = -0.04,
    fontsize: float = 8.5,
) -> None:
    """Draw task-name labels centered below each task block."""
    for task in present_tasks:
        start, end = blocks[task]
        ax.text(
            (start + end) / 2, y, TASK_LABELS[task],
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=fontsize,
        )
