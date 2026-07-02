"""Journal-quality matplotlib style for Scientific Data figures.

Mirrors the aesthetic of the companion fatigue-prediction figures
(Wong 2011 colorblind-safe palette, framed legends, subtle grid, boxed panel
badges, gradient CI ribbons).
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu

# ── Palettes ─────────────────────────────────────────────────────────────────
# Okabe–Ito / Wong 2011 colorblind-safe palette (doi:10.1038/nmeth.1618)
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#CC79A7",  # pink/purple
    "#009E73",  # teal/green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# Task-specific colors keyed by display name (same hues as PALETTE)
TASK_COLORS = {
    "Flat Trail": "#0072B2",
    "Stairs":     "#E69F00",
    "Ramp":       "#CC79A7",
    "Rest":       "#009E73",
}

# Borg-bin colours (5 bins: 1-2 → 9-10)
# Green (light fatigue) → yellow → orange → deep red (very hard)
BORG_BIN_PALETTE = ["#43A047", "#FDD835", "#FB8C00", "#E53935", "#B71C1C"]

# Continuous version of the same gradient, for plots that colour individual
# Borg ratings (1-10) directly instead of grouping them into bins.
BORG_CMAP = LinearSegmentedColormap.from_list("borg_fatigue", BORG_BIN_PALETTE)

MM_TO_INCH = 1 / 25.4


# ── rcParams ─────────────────────────────────────────────────────────────────
def apply_journal_style() -> None:
    """Apply Nature/Scientific Data publication defaults."""
    mpl.rcParams.update(
        {
            # Font
            "font.family":       "sans-serif",
            "font.sans-serif":   ["Arial", "Helvetica Neue", "DejaVu Sans"],
            "font.size":         9,
            "axes.titlesize":    10,
            "axes.titleweight":  "bold",
            "axes.labelsize":    9,
            "xtick.labelsize":   8,
            "ytick.labelsize":   8,
            "legend.fontsize":   8,
            # Legend — framed with light border
            "legend.frameon":    True,
            "legend.framealpha": 0.9,
            "legend.edgecolor":  "lightgray",
            "legend.fancybox":   False,
            # Axes
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.linewidth":    0.8,
            # Grid — subtle, both axes
            "axes.grid":         True,
            "grid.alpha":        0.25,
            "grid.linewidth":    0.5,
            "grid.color":        "#CCCCCC",
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size":  4,
            "ytick.major.size":  4,
            # DPI
            "figure.dpi":        150,
            "savefig.dpi":       300,
            "savefig.bbox":      "tight",
            "savefig.pad_inches": 0.05,
            # Embed fonts as TrueType in PDF/EPS — required by most journals
            "pdf.fonttype":      42,
            "ps.fonttype":       42,
        }
    )


# ── Utilities ─────────────────────────────────────────────────────────────────
def figure_size(n_cols: int = 1, aspect: float = 0.75) -> tuple[float, float]:
    """Return (width_in, height_in) for 1- or 2-column Nature layout.

    Args:
        n_cols: 1 → 89 mm wide; 2 → 183 mm wide.
        aspect: height / width ratio (default 0.75).
    """
    widths_mm = {1: 89, 2: 183}
    w_in = widths_mm.get(n_cols, 89) * MM_TO_INCH
    return w_in, w_in * aspect


def add_panel_label(ax: plt.Axes, label: str, x: float = 0.02, y: float = 0.97) -> None:
    """Bold boxed panel label (Nature/Science convention) at top-left inside axes.

    A white box with a light border ensures readability over any background.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="white",
            edgecolor="#AAAAAA",
            linewidth=0.8,
        ),
    )


def gradient_ci(
    ax: plt.Axes,
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str,
    n_layers: int = 8,
) -> None:
    """Layered fill_between to simulate a radial-fade CI ribbon.

    Outer layers are nearly transparent; inner layers are denser, giving a
    perceptually smooth gradient from the mean boundary outward.
    """
    for i in range(n_layers):
        frac = (i + 1) / n_layers
        alpha = 0.38 * (1 - frac ** 0.55)
        mid = (lo + hi) / 2.0
        ax.fill_between(
            x,
            mid - frac * (mid - lo),
            mid + frac * (hi - mid),
            alpha=alpha, color=color, linewidth=0,
        )


def fmt_pval(p: float) -> str:
    """Format p-value for figure annotations (Nature/APA convention)."""
    if p >= 0.001:
        return f"p = {p:.3f}"
    return "p < 0.001"


# ── Categorical comparison plot (violin + box + jitter + mean + sig bars) ───
# Shared by fig2_inter_task.py (Borg scale) and signal_transition_delta.py
# (onset-delta spike size) so both render with the same visual language.
_INK = "#000000"
_MUTED_INK = "#333333"

_VIOLIN_WIDTH = 0.68
_VIOLIN_ALPHA = 0.7
_VIOLIN_EDGE_WIDTH = 0.3

_BOX_WIDTH = 0.28
_BOX_ALPHA = 0.85
_BOX_EDGE_WIDTH = 0.3
_WHISKER_WIDTH = 0.7

_JITTER_HALF_WIDTH = 0.16
_JITTER_ALPHA = 0.3
_JITTER_SIZE = 10

_MEAN_MARKER_SIZE = 15
_MEAN_EDGE_WIDTH = 0.5

_SIG_LINE_WIDTH = 0.8
_SIG_TICK_DROP = 0.08   # length of the vertical ticks at each bracket end
_SIG_Y_PAD = 0.3         # gap above the tallest violin before the first bracket
_SIG_Y_STEP = 0.8        # vertical spacing between stacked brackets

_LEGEND_EDGE_WIDTH = 0.7
_LEGEND_PATCH_ALPHA = 0.78
_LEGEND_MARKER_SIZE = 6


def draw_violins(ax: plt.Axes, group_data: list, positions: list, colors: list) -> None:
    """Plain violin bodies — quartiles/whiskers come from the embedded boxplot."""
    vp = ax.violinplot(
        group_data, positions=positions, widths=_VIOLIN_WIDTH,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, color in zip(vp["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(_VIOLIN_ALPHA)
        pc.set_edgecolor(_INK)
        pc.set_linewidth(_VIOLIN_EDGE_WIDTH)


def draw_boxplots(ax: plt.Axes, group_data: list, positions: list, colors: list) -> None:
    """Narrow boxplot embedded inside each violin — quartiles, median, whiskers."""
    bp = ax.boxplot(
        group_data, positions=positions, widths=_BOX_WIDTH,
        patch_artist=True, showfliers=False, zorder=4,
    )
    for box_patch, color in zip(bp["boxes"], colors):
        box_patch.set_facecolor(color)
        box_patch.set_alpha(_BOX_ALPHA)
        box_patch.set_edgecolor(_INK)
        box_patch.set_linewidth(_BOX_EDGE_WIDTH)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set(color=_MUTED_INK, linewidth=_WHISKER_WIDTH)


def draw_jitter_points(ax: plt.Axes, group_data: list, positions: list, colors: list) -> None:
    """Jittered raw data points, colored per group, behind the boxplot/diamonds."""
    rng = np.random.RandomState(7)
    for pos, data, color in zip(positions, group_data, colors):
        jx = rng.uniform(pos - _JITTER_HALF_WIDTH, pos + _JITTER_HALF_WIDTH, len(data))
        ax.scatter(jx, data, color=color, alpha=_JITTER_ALPHA, s=_JITTER_SIZE,
                   zorder=2, ec="none")


def draw_mean_markers(ax: plt.Axes, group_data: list, positions: list, colors: list) -> None:
    """Color-matched mean diamonds, drawn on top of everything else."""
    means = [np.mean(v) for v in group_data]
    for pos, mean_val, color in zip(positions, means, colors):
        ax.scatter(pos, mean_val, marker="D", s=_MEAN_MARKER_SIZE, color=color,
                   edgecolor=_INK, linewidth=_MEAN_EDGE_WIDTH, zorder=10)


def draw_significance_bars(
    ax: plt.Axes,
    group_data: list,
    y_pad: float = _SIG_Y_PAD,
    y_step: float = _SIG_Y_STEP,
    tick_drop: float = _SIG_TICK_DROP,
    y_headroom: float = 0.5,
) -> float:
    """Pairwise Mann-Whitney U brackets for the first three group combinations.

    `y_pad`/`y_step`/`tick_drop`/`y_headroom` default to values tuned for a
    0-10 Borg scale; callers plotting a signal with a very different range
    (e.g. raw EDA in µS) should pass values scaled to their own data spread.

    Returns the y-limit needed to fit all drawn brackets.
    """
    pairs = [(0, 1), (1, 2), (0, 2)]
    y_top = max(
        np.percentile(v, 99) if len(v) > 0 else 0
        for v in group_data
    )
    y_start = y_top + y_pad
    for k, (i, j) in enumerate(pairs):
        if j >= len(group_data) or len(group_data[i]) < 2 or len(group_data[j]) < 2:
            continue
        _, p = mannwhitneyu(group_data[i], group_data[j], alternative="two-sided")
        y = y_start + k * y_step
        ax.plot([i, i, j, j], [y - tick_drop, y, y, y - tick_drop],
                color=_INK, linewidth=_SIG_LINE_WIDTH)
        ax.text((i + j) / 2, y + tick_drop * 0.75, fmt_pval(p),
                ha="center", va="bottom", fontsize=8)
    return y_start + len(pairs) * y_step + y_headroom


def draw_categorical_legend(
    ax: plt.Axes, present_order: list, colors: list, title: str = "Task type",
) -> None:
    """Group-color swatches plus a proxy handle for the mean diamond marker."""
    legend_handles = [
        mpatches.Patch(facecolor=color, alpha=_LEGEND_PATCH_ALPHA,
                       edgecolor=_INK, linewidth=_LEGEND_EDGE_WIDTH, label=t)
        for t, color in zip(present_order, colors)
    ]
    legend_handles.append(
        mlines.Line2D(
            [], [], marker="D", linestyle="none", markersize=_LEGEND_MARKER_SIZE,
            markerfacecolor="#FFFFFF", markeredgecolor=_INK,
            markeredgewidth=_LEGEND_EDGE_WIDTH, label="Mean",
        )
    )
    legend = ax.legend(handles=legend_handles, loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                        title=title, title_fontsize=8)
    legend.get_frame().set_linewidth(_LEGEND_EDGE_WIDTH)


def save_figure(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    """Save figure as 300 DPI PNG and vector PDF. Returns the PNG path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path


apply_journal_style()
