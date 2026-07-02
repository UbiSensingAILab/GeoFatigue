"""Builds lap-resampled elevation/EDA/pulse curves for the
elevation-physiology composite figure (fig6): folds each completed lap's
outbound+inbound legs into one 0-100% round-trip position axis (see
geofatigue.features.lap_segmentation), resamples elevation (from the lap's
own GPS trajectory) and per-minute baseline-z-scored EDA/pulse (matched by
timestamp) onto a shared 100-point grid, then averages across a
participant's own laps. Also computes the figure's caption statistics --
see docs/superpowers/specs/
2026-06-22-elevation-physiology-composite-figure-design.md section 4.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from shapely.geometry import LineString

from geofatigue.features.lap_segmentation import (
    TASK_START_DIRECTION,
    distance_from_start,
    find_completed_laps,
    lap_round_trip_pct,
    pick_extreme_endpoint,
)

DEFAULT_LAP_TOLERANCE_M = 2.0
DEFAULT_N_GRID_POINTS = 100
# Per-minute EDA/pulse baseline windows are short (session start -> first
# task start, typically 2-5 samples), so a participant's own baseline std
# can be tiny relative to their task-period signal, producing physiologically
# implausible z-scores (observed up to ~2000 on real data). Clipping bounds
# the per-lap z-scores before they get averaged across laps/participants,
# so a handful of unstable baselines can't dominate the population mean/SEM
# or distort the figure's y-axis scale.
DEFAULT_Z_CLIP = 10.0

LAP_PROFILE_COLUMNS = [
    "participant_id", "session_index", "task_label", "lap_index",
    "position_pct", "elevation_m", "eda_z", "pulse_z",
]
PARTICIPANT_TASK_PROFILE_COLUMNS = [
    "participant_id", "session_index", "task_label",
    "position_pct", "elevation_m", "eda_z", "pulse_z", "n_laps",
]

TASK_ORDER = ["flat_trail", "stairs", "ramp"]


def _nearest_value_at(timestamps_us: np.ndarray, values: np.ndarray, query_us: np.ndarray) -> np.ndarray:
    """Nearest-in-time lookup of `values` at each `query_us` timestamp, via
    searchsorted (timestamps_us must be sorted ascending). Used to attach
    per-minute EDA/pulse z-scores (sparse, ~1/minute) to GPS-rate (~1 Hz)
    arc-length samples within a lap. Returns all-NaN if `timestamps_us` is
    empty (no biomarker coverage for this task)."""
    if len(timestamps_us) == 0:
        return np.full(len(query_us), np.nan)
    idx = np.searchsorted(timestamps_us, query_us)
    idx = np.clip(idx, 1, len(timestamps_us) - 1)
    left = timestamps_us[idx - 1]
    right = timestamps_us[idx]
    use_left = np.abs(query_us - left) <= np.abs(query_us - right)
    return np.where(use_left, values[idx - 1], values[idx])


def build_participant_task_lap_profiles(
    participant_id: str,
    session_index: int,
    task_label: str,
    gps_trajectory: pd.DataFrame,
    centerline_3857: LineString,
    eda_z_df: pd.DataFrame,
    pulse_z_df: pd.DataFrame,
    task_start_time: pd.Timestamp,
    tolerance_m: float = DEFAULT_LAP_TOLERANCE_M,
    n_grid_points: int = DEFAULT_N_GRID_POINTS,
    z_clip: float = DEFAULT_Z_CLIP,
    lidar_elevation_profile: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> pd.DataFrame:
    """Per-lap resampled elevation/EDA/pulse curves for one participant's
    one task.

    Args:
        gps_trajectory: One task's cleaned, map-matched GPS trajectory, from
            geofatigue.loaders.lap_trajectory.build_task_lap_trajectory --
            indexed by timestamp, columns {x, y, altitude, arc_length_m}.
        centerline_3857: Same centerline used to build `gps_trajectory`
            (EPSG:3857) -- used here only to locate the lap start point.
        eda_z_df / pulse_z_df: This session's baseline-z-scored per-task
            biomarker records, from
            geofatigue.features.biomarker_baseline.build_session_biomarker_records_baseline_z,
            already filtered to task_label by the caller -- columns
            minutes_since_task_start, value_z.
        task_start_time: This task's actual start_time from
            session_metadata.json -- the anchor eda_z_df/pulse_z_df's
            minutes_since_task_start are relative to. Must NOT be inferred
            from gps_trajectory's own first timestamp, which can start
            slightly before/after the metadata task window.
        z_clip: Clip eda_z/pulse_z to [-z_clip, z_clip] (see DEFAULT_Z_CLIP)
            before they're returned, so an unstable per-participant baseline
            can't dominate the population mean/SEM downstream. NaN values
            (no biomarker coverage at that point) pass through unclipped.
        lidar_elevation_profile: Optional (position_pct, elevation_m) from
            geofatigue.loaders.tif_elevation.sample_centerline_elevation_profile.
            When provided, uses survey-grade LiDAR terrain elevation instead of
            each lap's noisy GPS altitude. The same profile is applied to every
            lap (it is the physical terrain, not a per-lap measurement).
            Falls back to GPS altitude when None.

    Returns:
        DataFrame with columns LAP_PROFILE_COLUMNS -- one row per (lap_index,
        position_pct grid point). Empty (no rows) if the GPS trajectory has
        no completed laps.
    """
    if gps_trajectory.empty:
        return pd.DataFrame(columns=LAP_PROFILE_COLUMNS)

    start_point = pick_extreme_endpoint(centerline_3857, TASK_START_DIRECTION[task_label])
    dist = distance_from_start(
        centerline_3857, start_point,
        gps_trajectory["x"].to_numpy(), gps_trajectory["y"].to_numpy(),
    )
    laps = find_completed_laps(gps_trajectory.index, dist, tolerance_m=tolerance_m)
    if not laps:
        return pd.DataFrame(columns=LAP_PROFILE_COLUMNS)

    task_start_us = task_start_time.value // 1_000
    eda_ts_us = task_start_us + (eda_z_df["minutes_since_task_start"].to_numpy() * 60_000_000.0)
    eda_vals = eda_z_df["value_z"].to_numpy()
    pulse_ts_us = task_start_us + (pulse_z_df["minutes_since_task_start"].to_numpy() * 60_000_000.0)
    pulse_vals = pulse_z_df["value_z"].to_numpy()

    grid_pct = np.linspace(0, 100, n_grid_points)
    rows = []
    for lap_index, (lap_start, lap_end) in enumerate(laps):
        lap = gps_trajectory[(gps_trajectory.index >= lap_start) & (gps_trajectory.index < lap_end)]
        if len(lap) < 2:
            continue
        position_pct = lap_round_trip_pct(lap["arc_length_m"].to_numpy())

        lap_ts_us = lap.index.values.astype("datetime64[us]").astype(np.int64)
        eda_at_lap = _nearest_value_at(eda_ts_us, eda_vals, lap_ts_us)
        pulse_at_lap = _nearest_value_at(pulse_ts_us, pulse_vals, lap_ts_us)

        if lidar_elevation_profile is not None:
            ref_pct, ref_elev = lidar_elevation_profile
            elevation_grid = np.interp(grid_pct, ref_pct, ref_elev)
        else:
            elevation_grid = np.interp(grid_pct, position_pct, lap["altitude"].to_numpy())
        eda_grid = np.clip(np.interp(grid_pct, position_pct, eda_at_lap), -z_clip, z_clip)
        pulse_grid = np.clip(np.interp(grid_pct, position_pct, pulse_at_lap), -z_clip, z_clip)

        for pct, elev, eda_z, pulse_z in zip(grid_pct, elevation_grid, eda_grid, pulse_grid):
            rows.append({
                "participant_id": participant_id,
                "session_index": session_index,
                "task_label": task_label,
                "lap_index": lap_index,
                "position_pct": float(pct),
                "elevation_m": float(elev),
                "eda_z": float(eda_z),
                "pulse_z": float(pulse_z),
            })

    return pd.DataFrame(rows, columns=LAP_PROFILE_COLUMNS)


def average_laps_per_participant(lap_profile_df: pd.DataFrame) -> pd.DataFrame:
    """Average a participant's own completed laps together (spec step 2),
    plus how many laps contributed (for the figure caption's mean +- SD
    laps-per-participant statistic).

    Args:
        lap_profile_df: Output of build_participant_task_lap_profiles,
            concatenated across participants/sessions/tasks.

    Returns:
        DataFrame with columns PARTICIPANT_TASK_PROFILE_COLUMNS -- one row
        per (participant_id, session_index, task_label, position_pct).
    """
    if lap_profile_df.empty:
        return pd.DataFrame(columns=PARTICIPANT_TASK_PROFILE_COLUMNS)

    group_keys = ["participant_id", "session_index", "task_label", "position_pct"]
    n_laps = (
        lap_profile_df.groupby(group_keys[:-1])["lap_index"].nunique()
        .rename("n_laps").reset_index()
    )
    averaged = (
        lap_profile_df.groupby(group_keys)[["elevation_m", "eda_z", "pulse_z"]]
        .mean().reset_index()
    )
    merged = averaged.merge(n_laps, on=group_keys[:-1], how="left")
    return merged[PARTICIPANT_TASK_PROFILE_COLUMNS]


def _holm_bonferroni(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction across `pvalues`. NaN entries
    (too few paired participants to test, see build_caption_stats) are left
    as NaN and excluded from the correction family -- they were never a
    real test, so they shouldn't count against alpha or be corrected."""
    valid = [(i, p) for i, p in enumerate(pvalues) if not np.isnan(p)]
    corrected = list(pvalues)
    m = len(valid)
    if m == 0:
        return corrected
    running_max = 0.0
    for rank, (i, p) in enumerate(sorted(valid, key=lambda t: t[1])):
        running_max = max(running_max, p * (m - rank))
        corrected[i] = min(running_max, 1.0)
    return corrected


def build_caption_stats(lap_profile_df: pd.DataFrame) -> Dict[str, dict]:
    """Per-task caption statistics: n participants/laps, mean +- SD
    completed laps per participant, the 95th percentile of |EDA/pulse
    z-score|, and an ascent (0-50%) vs. descent (50-100%) paired t-test per
    signal -- see design spec section 4.

    The 95th percentile (not max) is reported deliberately: eda_z/pulse_z
    are clipped to +-DEFAULT_Z_CLIP in build_participant_task_lap_profiles
    to keep one unstable baseline from dominating the population mean/SEM,
    which means the raw max is frequently just that clip ceiling rather
    than a real measured value. The 95th percentile is far less sensitive
    to a handful of still-clipped participants. The ascent/descent p-values
    are Holm-Bonferroni-corrected across all tasks/signals jointly (up to 6
    tests at once), since this is the family of comparisons this caption
    actually reports.

    Only the first (chronologically earliest) session per participant is
    used, matching the rest of this figure's pooling.

    Returns:
        Dict keyed by task_label, each value a dict with keys:
        n_participants, n_laps, laps_per_participant_mean,
        laps_per_participant_sd, eda_z_p95, pulse_z_p95,
        eda_ascent_descent_p, pulse_ascent_descent_p (the latter two
        Holm-Bonferroni-corrected).
    """
    stats: Dict[str, dict] = {}
    if lap_profile_df.empty:
        return stats

    curves_df = average_laps_per_participant(lap_profile_df)
    first_session = curves_df.groupby("participant_id")["session_index"].min()
    curves_df = curves_df[curves_df["session_index"] == curves_df["participant_id"].map(first_session)]

    lap_counts = (
        lap_profile_df.drop_duplicates(["participant_id", "session_index", "task_label", "lap_index"])
        .groupby(["participant_id", "session_index", "task_label"]).size()
        .rename("n_laps").reset_index()
    )
    lap_counts = lap_counts[lap_counts["session_index"] == lap_counts["participant_id"].map(first_session)]

    raw_pvalues: List[float] = []
    pvalue_slots: List[tuple] = []

    for task in TASK_ORDER:
        task_curves = curves_df[curves_df["task_label"] == task]
        if task_curves.empty:
            continue
        task_laps = lap_counts[lap_counts["task_label"] == task]["n_laps"]

        ascent = task_curves[task_curves["position_pct"] <= 50]
        descent = task_curves[task_curves["position_pct"] > 50]
        ascent_eda = ascent.groupby("participant_id")["eda_z"].mean()
        descent_eda = descent.groupby("participant_id")["eda_z"].mean()
        common = ascent_eda.index.intersection(descent_eda.index)
        eda_p = float(ttest_rel(ascent_eda[common], descent_eda[common]).pvalue) if len(common) >= 2 else float("nan")

        ascent_pulse = ascent.groupby("participant_id")["pulse_z"].mean()
        descent_pulse = descent.groupby("participant_id")["pulse_z"].mean()
        common_p = ascent_pulse.index.intersection(descent_pulse.index)
        pulse_p = (
            float(ttest_rel(ascent_pulse[common_p], descent_pulse[common_p]).pvalue)
            if len(common_p) >= 2 else float("nan")
        )

        stats[task] = {
            "n_participants": int(task_curves["participant_id"].nunique()),
            "n_laps": int(task_laps.sum()),
            "laps_per_participant_mean": float(task_laps.mean()) if len(task_laps) else float("nan"),
            "laps_per_participant_sd": float(task_laps.std()) if len(task_laps) > 1 else 0.0,
            "eda_z_p95": float(task_curves["eda_z"].abs().quantile(0.95)),
            "pulse_z_p95": float(task_curves["pulse_z"].abs().quantile(0.95)),
        }
        raw_pvalues.extend([eda_p, pulse_p])
        pvalue_slots.extend([(task, "eda_ascent_descent_p"), (task, "pulse_ascent_descent_p")])

    for (task, key), p in zip(pvalue_slots, _holm_bonferroni(raw_pvalues)):
        stats[task][key] = p
    return stats
