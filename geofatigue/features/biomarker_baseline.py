"""Baseline z-scoring for Empatica's per-minute aggregated digital-biomarker
signals (EDA, pulse-rate, ...), mirroring geofatigue.features.eda's
within-session baseline-correction approach but without the
AVRO-sampling-rate-specific Hampel/low-pass filtering step -- the
aggregated_per_minute export is already a per-minute summary, not a raw
high-frequency stream, so there is nothing left to despike/smooth.
"""
from typing import Dict

import numpy as np
import pandas as pd

from geofatigue.features.eda import compute_baseline_stats, compute_baseline_window, zscore

_RECORD_COLUMNS = [
    "participant_id", "session_index", "task_label",
    "minutes_since_task_start", "value_z",
]


def build_session_biomarker_records_baseline_z(
    participant_id: str,
    session_index: int,
    session: Dict,
    biomarker_df: pd.DataFrame,
    value_column: str,
    log_transform: bool = False,
) -> pd.DataFrame:
    """Baseline-z-score one session's per-minute biomarker signal, sliced
    per task.

    Baseline = session start until the earliest task start (see
    geofatigue.features.eda.compute_baseline_window); the same mean/std is
    used to z-score every task in the session, exactly mirroring how
    build_session_eda_records baseline-corrects the raw AVRO EDA stream.

    Args:
        biomarker_df: DataFrame with columns timestamp_us, <value_column>
            (as returned by
            geofatigue.loaders.digital_biomarkers_csv.load_participant_biomarker_csv).
        value_column: Name of the signal's value column, e.g.
            'eda_scl_usiemens', 'pulse_rate_bpm'.
        log_transform: Apply log1p before computing the baseline mean/std and
            z-scoring. Skin-conductance level (EDA) is right-skewed and
            heteroscedastic across rest vs. activity -- on real data, a
            handful of minutes' resting baseline can have near-zero variance,
            which makes the raw z = (x - mean) / std ratio blow up to
            implausible magnitudes (observed up to ~7000 on this dataset's
            EDA) once the task period departs from that near-zero-variance
            baseline. Log-transforming first is the standard remedy for this
            in EDA literature (e.g. Dawson, Schell & Filion's EDA recording
            guidelines) and is why this is signal-specific: pulse rate (bpm)
            doesn't have the same skew/heteroscedasticity and should use
            log_transform=False (the default).

    Returns:
        Tidy DataFrame with columns: participant_id, session_index,
        task_label, minutes_since_task_start, value_z.
        Returns an empty DataFrame (no rows) if the session has no
        overlapping samples at all. Raises ValueError (propagated from
        compute_baseline_stats/compute_baseline_window) if the session has
        no tasks, no baseline-window coverage, or a zero-variance baseline
        -- callers iterating many sessions should catch this per-session,
        the same way scripts/generate_figures.py's build_eda_profile_df
        already does for build_session_eda_records.
    """
    session_start_us = pd.Timestamp(session["start_time"], tz="UTC").value // 1_000
    session_end_us = pd.Timestamp(session["end_time"], tz="UTC").value // 1_000
    session_df = biomarker_df[
        (biomarker_df["timestamp_us"] >= session_start_us)
        & (biomarker_df["timestamp_us"] < session_end_us)
    ]
    if session_df.empty:
        return pd.DataFrame(columns=_RECORD_COLUMNS)

    ts = session_df["timestamp_us"].to_numpy()
    vals = session_df[value_column].to_numpy()
    if log_transform:
        vals = np.log1p(np.clip(vals, 0.0, None))

    baseline_start, baseline_end = compute_baseline_window(session)
    mean, std = compute_baseline_stats(ts, vals, baseline_start, baseline_end)
    z = zscore(vals, mean, std)

    rows = []
    for task in session["tasks"]:
        t_start_us = pd.Timestamp(task["start_time"], tz="UTC").value // 1_000
        t_end_us = pd.Timestamp(task["end_time"], tz="UTC").value // 1_000
        task_mask = (ts >= t_start_us) & (ts < t_end_us)
        if not task_mask.any():
            continue
        minutes = (ts[task_mask] - t_start_us) / 1_000_000.0 / 60.0
        for m, zv in zip(minutes, z[task_mask]):
            rows.append({
                "participant_id": participant_id,
                "session_index": session_index,
                "task_label": task["label"],
                "minutes_since_task_start": float(m),
                "value_z": float(zv),
            })

    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)


def build_session_biomarker_records_per_task_baseline_z(
    participant_id: str,
    session_index: int,
    session: Dict,
    biomarker_df: pd.DataFrame,
    value_column: str,
    log_transform: bool = False,
) -> pd.DataFrame:
    """Like build_session_biomarker_records_baseline_z but uses the period
    immediately before each task as that task's individual baseline:
    - First task:       session_start → task[0].start_time  (pre-session rest)
    - Subsequent tasks: prior_task.end_time → task.start_time  (inter-task break)

    This normalises each task independently, removing cross-task arousal
    carry-over so task-intrinsic EDA response shapes are directly comparable
    (used by Fig 14). Tasks whose preceding break contains no samples or has
    zero variance are silently skipped rather than raising.
    """
    session_start_us = pd.Timestamp(session["start_time"], tz="UTC").value // 1_000
    session_end_us = pd.Timestamp(session["end_time"], tz="UTC").value // 1_000
    session_df = biomarker_df[
        (biomarker_df["timestamp_us"] >= session_start_us)
        & (biomarker_df["timestamp_us"] < session_end_us)
    ]
    if session_df.empty:
        return pd.DataFrame(columns=_RECORD_COLUMNS)

    ts = session_df["timestamp_us"].to_numpy()
    vals = session_df[value_column].to_numpy()
    if log_transform:
        vals = np.log1p(np.clip(vals, 0.0, None))

    tasks_sorted = sorted(session["tasks"], key=lambda t: t["start_time"])
    session_start_ts = pd.Timestamp(session["start_time"], tz="UTC")

    rows = []
    for i, task in enumerate(tasks_sorted):
        t_start_ts = pd.Timestamp(task["start_time"], tz="UTC")
        t_end_ts = pd.Timestamp(task["end_time"], tz="UTC")
        t_start_us = t_start_ts.value // 1_000
        t_end_us = t_end_ts.value // 1_000

        if i == 0:
            baseline_start = session_start_ts
            baseline_end = t_start_ts
        else:
            prior = tasks_sorted[i - 1]
            baseline_start = pd.Timestamp(prior["end_time"], tz="UTC")
            baseline_end = t_start_ts

        try:
            mean, std = compute_baseline_stats(ts, vals, baseline_start, baseline_end)
        except ValueError:
            continue

        task_mask = (ts >= t_start_us) & (ts < t_end_us)
        if not task_mask.any():
            continue

        z_task = zscore(vals[task_mask], mean, std)
        minutes = (ts[task_mask] - t_start_us) / 1_000_000.0 / 60.0
        for m, zv in zip(minutes, z_task):
            rows.append({
                "participant_id": participant_id,
                "session_index": session_index,
                "task_label": task["label"],
                "minutes_since_task_start": float(m),
                "value_z": float(zv),
            })

    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)
