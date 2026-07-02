"""Fig6: Elevation-physiology composite -- EDA, baseline
z-scored and lap-averaged, plotted against each task's literal elevation
profile on one combined panel.

Five fixed-width x-axis blocks in real task order (flat trail, break,
stairs, break, ramp); each task block is schematic (the lap-averaged 0-100%
curve stretched to fill it, not literally time-scaled), break blocks are
small and cross-hatched with no curve. 
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geofatigue.features.lap_profile import average_laps_per_participant, build_caption_stats
from geofatigue.figures._block_layout import (
    TASK_ORDER, TASK_LABELS,
    block_layout, task_x, draw_break_blocks,
)
from geofatigue.figures.style import PALETTE, figure_size, fmt_pval

ELEVATION_COLOR = PALETTE[4]  # sky blue
EDA_COLOR = PALETTE[5]        # vermillion
PULSE_COLOR = PALETTE[7]


def _fmt_p_or_na(p: float) -> str:
    """fmt_pval assumes a real p-value; an ascent/descent comparison with
    too few paired participants reports NaN (see
    geofatigue.features.lap_profile.build_caption_stats), which fmt_pval
    would otherwise silently mis-render as 'p < 0.001'."""
    if np.isnan(p):
        return "p = n/a"
    return fmt_pval(p)


def _format_caption(stats: Dict[str, dict]) -> str:
    """One line per task: n participants/laps, mean +- SD laps per
    participant, and EDA/pulse 95th-percentile |z| with their
    ascent-vs-descent paired-t-test p-value -- see design spec section 4.

    The 95th percentile (not max/"peak") is reported because eda_z/pulse_z
    are clipped before this point (see
    geofatigue.features.lap_profile.DEFAULT_Z_CLIP) -- a max would often
    just report that clip ceiling rather than a real measured value.
    p-values are Holm-Bonferroni-corrected across all tasks/signals jointly
    by build_caption_stats.

    Args:
        stats: Output of geofatigue.features.lap_profile.build_caption_stats.
    """
    lines = []
    for task in TASK_ORDER:
        s = stats.get(task)
        if s is None:
            continue
        lines.append(
            f"{TASK_LABELS[task]}: n={s['n_participants']} participants, {s['n_laps']} laps "
            f"({s['laps_per_participant_mean']:.1f} ± {s['laps_per_participant_sd']:.1f} per participant). "
            f"EDA |z| (95th pct) = {s['eda_z_p95']:.2f} (ascent vs. descent, Holm-corrected, "
            f"{_fmt_p_or_na(s['eda_ascent_descent_p'])}); "
            f"pulse |z| (95th pct) = {s['pulse_z_p95']:.2f} (ascent vs. descent, Holm-corrected, "
            f"{_fmt_p_or_na(s['pulse_ascent_descent_p'])})."
        )
    return "\n".join(lines)


def _population_curve(
    df: pd.DataFrame, task_label: str, value_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean +- SEM of `value_column` across participants, at each
    position_pct, for one task. Returns (position_pct, mean, sem, n)."""
    task_df = df[df["task_label"] == task_label]
    pivot = task_df.pivot(index="participant_id", columns="position_pct", values=value_column)
    pivot = pivot.dropna(how="all")
    mean = pivot.mean(axis=0).to_numpy()
    n = len(pivot)
    if n > 1:
        sem = (pivot.std(axis=0, ddof=1) / np.sqrt(n)).to_numpy()
    else:
        sem = np.zeros_like(mean)
    return pivot.columns.to_numpy(dtype=float), mean, sem, n


def plot_elevation_physiology_composite(
    lap_profile_df: pd.DataFrame,
    task_distances_m: "dict | None" = None,
) -> plt.Figure:
    """Single combined-panel figure: elevation backdrop + EDA mean+-SEM
    bands (plus thin per-participant lines) across the three tasks, laid out
    as five blocks (task, break, task, break, task).

    Args:
        lap_profile_df: Output of
            geofatigue.features.lap_profile.build_participant_task_lap_profiles,
            concatenated across participants/tasks (one row per lap per grid
            point) -- columns geofatigue.features.lap_profile.LAP_PROFILE_COLUMNS.
            Averaged across each participant's own laps internally via
            average_laps_per_participant before any population-level
            aggregation.
        task_distances_m: Optional dict mapping task label to one-way
            centerline length in metres.  When provided, each task block's
            x-axis width is proportional to path distance.  When None, all
            task blocks have equal width.

    Returns:
        matplotlib Figure with one axes (elevation) plus one twinned axis
        (EDA).
    """
    _w, _h = figure_size(2, aspect=0.52)
    fig, ax_elev = plt.subplots(figsize=(_w * 1.25, _h))

    if lap_profile_df.empty:
        ax_elev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_elev.transAxes)
        return fig

    curves_df = average_laps_per_participant(lap_profile_df)
    first_session = curves_df.groupby("participant_id")["session_index"].min()
    curves_df = curves_df[curves_df["session_index"] == curves_df["participant_id"].map(first_session)]

    present_tasks = [t for t in TASK_ORDER if t in curves_df["task_label"].values]
    if not present_tasks:
        ax_elev.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_elev.transAxes)
        return fig

    elevation_curves = {t: _population_curve(curves_df, t, "elevation_m") for t in present_tasks}
    # Normalise to a common ground-level reference (the global elevation
    # minimum across all tasks).  Per-task lap-start normalisation would
    # collapse the ramp (which starts at the top of the stairs structure)
    # to the same apparent baseline as the flat trail and stairs, hiding
    # the real elevation difference between task starting points.
    _all_elev = np.concatenate([mean for _, mean, _, _ in elevation_curves.values()])
    _global_min = float(_all_elev.min())
    elevation_curves = {
        t: (pct, mean - _global_min, sem, n)
        for t, (pct, mean, sem, n) in elevation_curves.items()
    }
    elevation_floor = 0.0

    # Compute block layout once; pass explicitly to helpers.
    blocks = block_layout(task_distances_m)

    ax_z = ax_elev.twinx()
    draw_break_blocks(ax_elev, blocks)

    # Padding between the elevation fill and the adjacent break-block edges
    # so the exact start/end of each task's elevation profile is visible.
    _ELEV_PAD = 0.005

    eda_handle = pulse_handle = None
    for task in present_tasks:
        pct, elev_mean, _, _ = elevation_curves[task]
        x = task_x(task, pct, blocks, padding=_ELEV_PAD)
        ax_elev.fill_between(x, elev_mean, elevation_floor, color=ELEVATION_COLOR, alpha=0.4, zorder=2)

        pct_e, eda_mean, eda_sem, _ = _population_curve(curves_df, task, "eda_z")
        x_e = task_x(task, pct_e, blocks, padding=_ELEV_PAD)
        (eda_handle,) = ax_z.plot(x_e, eda_mean, color=EDA_COLOR, linewidth=3.0, zorder=5)
        ax_z.fill_between(x_e, eda_mean - eda_sem, eda_mean + eda_sem, color=EDA_COLOR, alpha=0.25, zorder=4)

    for task in present_tasks:
        start, end = blocks[task]
        ax_elev.text(
            (start + end) / 2, -0.04, TASK_LABELS[task], transform=ax_elev.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5,
        )

    ax_elev.set_xlim(0, blocks["ramp"][1])
    ax_elev.set_xticks([])
    ax_elev.set_ylabel("Elevation above lowest point (m)", color="#333333", labelpad=6)
    ax_z.set_ylabel("Z-score (EDA)", color="#333333", labelpad=6)
    ax_elev.set_title("Elevation Profile vs. Physiological Response by Task", pad=8)

    if eda_handle is not None and pulse_handle is not None:
        fig.legend(
            [eda_handle, pulse_handle], ["EDA (mean ± SEM)"],
            loc="upper right", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8, frameon=False,
        )

    # caption = _format_caption(build_caption_stats(lap_profile_df))
    # n_caption_lines = caption.count("\n") + 1 if caption else 0
    # if caption:
    #     fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=6.5, color="#333333")

    top_margin = 0.95
    bottom_margin = 0.12
    # bottom_margin = 0.12 + 0.04 * n_caption_lines
    fig.tight_layout(rect=[0, bottom_margin, 1, top_margin])
    return fig