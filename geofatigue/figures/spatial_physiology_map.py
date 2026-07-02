"""Spatial physiology heatmap — EDA relative to each participant's personal
baseline, projected onto the experiment-site satellite map via GPS (Fig 7).

Hexagonal spatial bins are colored by the MEAN z-score of EDA readings inside
them (each reading standardized against that participant's own mean/std first).
Pooling raw values across many participants mixes each person's baseline offset
in with the spatial effect — standardizing first removes that offset. A one-way
ANOVA across zones is annotated as a quantitative check on the visual pattern.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import contextily as cx
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from geofatigue.figures.style import fmt_pval

# Short on-image credit, drawn in the bottom-right corner.
# The full attribution (cx.providers.Esri.WorldImagery["attribution"])
# should still appear in the manuscript's figure caption.
_ESRI_ATTRIBUTION = "Imagery © Esri"

# Diverging colormap centered at 0 (= participant's own baseline).
# Red = above baseline, blue = below.
HEATMAP_CMAP = "RdBu_r"

# Zone-boundary outline colors keyed by zone name.
ZONE_OUTLINE_COLORS = {
    "resting_area": "#2a41bb",
    "stairs":       "#48784b",
    "flat_trail":   "#fb7d6c",
    "ramp":         "#734791",
}


def _empty_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="#666666")
    ax.set_axis_off()
    return fig


def _to_web_mercator(df: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=3857)


def _draw_zone_outlines(ax: plt.Axes, zone_polygons: gpd.GeoDataFrame) -> None:
    zones_3857 = zone_polygons.to_crs(epsg=3857)
    for zone_name, group in zones_3857.groupby("zone"):
        color = ZONE_OUTLINE_COLORS.get(zone_name, "white")
        group.plot(ax=ax, color=color, alpha=0.38, zorder=4)
        group.boundary.plot(ax=ax, color=color, linewidth=2.5, zorder=5)


def _zone_summary_text(sub: pd.DataFrame) -> str:
    """Per-zone mean z-score + n, in ZONE_OUTLINE_COLORS order, for figure captions."""
    zone_stats = sub.groupby("zone")["value_z"].agg(["mean", "count"])
    zone_order = [z for z in ZONE_OUTLINE_COLORS if z in zone_stats.index]
    return ", ".join(
        f"{zone}: {zone_stats.loc[zone, 'mean']:+.2f} (n={int(zone_stats.loc[zone, 'count'])})"
        for zone in zone_order
    )


def _annotate_anova(ax: plt.Axes, sub: pd.DataFrame) -> None:
    """One-way ANOVA of value_z across zones — annotated in the top-right corner."""
    groups = [g["value_z"].to_numpy() for _, g in sub.groupby("zone") if len(g) >= 2]
    if len(groups) < 2:
        return
    _, p = f_oneway(*groups)
    ax.text(
        0.98, 0.98, f"ANOVA across zones\n{fmt_pval(p)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        zorder=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             edgecolor="lightgray", alpha=0.92),
    )


def _add_satellite_basemap(ax: plt.Axes) -> None:
    """Esri World Imagery basemap — free for academic use, no API key required."""
    cx.add_basemap(
        ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857",
        zorder=-1, attribution=False,
    )
    ax.text(
        0.995, 0.005, _ESRI_ATTRIBUTION, transform=ax.transAxes,
        ha="right", va="bottom", size=10, zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
    )


def plot_spatial_physiology_heatmap(
    spatial_df: pd.DataFrame, zone_polygons: gpd.GeoDataFrame,
    gridsize: int = 20, min_count: int = 3,
) -> plt.Figure:
    """Fig 7 — hexbin heatmap of EDA z-score over the experiment site map.

    Each spatial bin is colored by the MEAN z-score (standardized against each
    participant's own mean/std) of EDA readings inside it. Bins with fewer than
    `min_count` readings are left blank to suppress noise. The color scale is
    symmetric and diverging, centered at 0 (= each participant's personal
    baseline), sized to the 98th percentile of binned means present.

    Zone-boundary outlines and an Esri World Imagery satellite basemap are
    drawn beneath the hexbins.

    Args:
        spatial_df: Long-format DataFrame with columns {participant_id, signal,
            value, value_z, timestamp_us, latitude, longitude, zone} — see
            geofatigue.loaders.spatial_physiology.build_spatial_biomarker_records.
        zone_polygons: GeoDataFrame with columns {zone, geometry} (EPSG:4326)
            — see geofatigue.loaders.spatial_physiology.load_zone_polygons.
        gridsize: Number of hexagons across the x-axis (passed to hexbin).
        min_count: Minimum readings per bin; bins below this are left blank.

    Returns:
        matplotlib Figure.
    """
    eda_df = spatial_df[spatial_df["signal"] == "eda"]
    if eda_df.empty:
        return _empty_figure("No spatially-located EDA data available")

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.5))
    gdf = _to_web_mercator(eda_df)

    ax.set_aspect("equal")
    hb = ax.hexbin(
        gdf.geometry.x, gdf.geometry.y, C=gdf["value_z"], reduce_C_function=np.mean,
        gridsize=15, cmap=HEATMAP_CMAP, mincnt=min_count, alpha=0.95,
        linewidths=0.2, zorder=6,
    )

    non_empty = hb.get_array().compressed()
    vmax = float(np.nanpercentile(np.abs(non_empty), 98)) if non_empty.size else 1.0
    vmax = max(vmax, 1e-6)
    hb.set_clim(-vmax, vmax)

    zones_no_rest = zone_polygons[zone_polygons["zone"] != "resting_area"]
    _draw_zone_outlines(ax, zones_no_rest)
    _annotate_anova(ax, eda_df)
    _add_satellite_basemap(ax)

    cbar = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("EDA — z-score relative to personal mean")
    cbar.ax.yaxis.label.set_size(11)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        "Spatial Heatmap of EDA Relative to Each Participant's Baseline",
        fontsize=10, fontweight="bold", pad=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
