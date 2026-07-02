"""Pure functions for EDA baseline correction and within-subject z-scoring.

Preprocessing follows common ambulatory-EDA practice: a Hampel filter
removes motion-artifact spikes (this dataset involves walking/stairs/ramp
tasks), then a zero-phase Butterworth low-pass smooths the signal, before
baseline z-scoring against each session's initial resting period.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from geofatigue.filters.signal_filter import hampel, lowpass

EDA_HAMPEL_WINDOW = 5
EDA_HAMPEL_K_SIGMA = 1.4826
EDA_LOWPASS_CUTOFF_HZ = 1.0  # must stay below Nyquist (fs/2) for real ~4 Hz EDA
EDA_LOWPASS_ORDER = 4

_RECORD_COLUMNS = [
    "participant_id", "session_index", "task_label",
    "minutes_since_task_start", "eda_z",
]


def filter_eda_signal(values: np.ndarray, fs: float) -> np.ndarray:
    """Hampel spike removal followed by zero-phase low-pass smoothing."""
    despiked = hampel(values, window_size=EDA_HAMPEL_WINDOW, k_sigma=EDA_HAMPEL_K_SIGMA)
    return lowpass(despiked, fs=fs, cutoff_hz=EDA_LOWPASS_CUTOFF_HZ, order=EDA_LOWPASS_ORDER)


def compute_baseline_window(session: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Baseline = session start until the start of the earliest task.

    Tasks are sorted by start_time so an out-of-order tasks list still
    yields the correct (earliest) boundary.
    """
    if not session["tasks"]:
        raise ValueError("Session has no tasks; cannot determine baseline end.")
    session_start = pd.Timestamp(session["start_time"], tz="UTC")
    task_starts = [pd.Timestamp(t["start_time"], tz="UTC") for t in session["tasks"]]
    return session_start, min(task_starts)


def compute_baseline_stats(
    timestamps_us: np.ndarray,
    values: np.ndarray,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
) -> Tuple[float, float]:
    """Mean and std of `values` whose timestamp falls in [baseline_start, baseline_end).

    Raises ValueError if no samples fall in the window, or if std == 0
    (z-scoring would divide by zero).
    """
    start_us = baseline_start.value // 1_000
    end_us = baseline_end.value // 1_000
    mask = (timestamps_us >= start_us) & (timestamps_us < end_us)
    baseline_values = values[mask]
    if baseline_values.size == 0:
        raise ValueError("No samples found in baseline window.")
    mean = float(np.mean(baseline_values))
    std = float(np.std(baseline_values))
    if std == 0:
        raise ValueError("Baseline std is zero; cannot z-score.")
    return mean, std


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """(values - mean) / std. Raises ValueError if std <= 0."""
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    return (values - mean) / std


def build_session_eda_records(
    participant_id: str,
    session_index: int,
    session: Dict,
    eda_timestamps_us: np.ndarray,
    eda_values: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    """Filter + baseline-z-score one session's EDA, sliced per task.

    Filtering is applied once over the full continuous session window
    (not per task) to avoid boundary artifacts the zero-phase low-pass
    would otherwise introduce at each short task segment's edges.

    Returns a tidy DataFrame with columns:
        participant_id, session_index, task_label,
        minutes_since_task_start, eda_z
    Tasks with no overlapping EDA samples are silently skipped (no rows
    emitted) rather than raising, since a participant's recorded signal
    may not cover every task in the metadata.
    """
    session_start_us = pd.Timestamp(session["start_time"], tz="UTC").value // 1_000
    session_end_us = pd.Timestamp(session["end_time"], tz="UTC").value // 1_000

    session_mask = (eda_timestamps_us >= session_start_us) & (eda_timestamps_us < session_end_us)
    ts = eda_timestamps_us[session_mask]
    vals = eda_values[session_mask]
    if ts.size == 0:
        return pd.DataFrame(columns=_RECORD_COLUMNS)

    filtered = filter_eda_signal(vals, fs=fs)

    baseline_start, baseline_end = compute_baseline_window(session)
    mean, std = compute_baseline_stats(ts, filtered, baseline_start, baseline_end)
    z = zscore(filtered, mean, std)

    rows = []
    for task in session["tasks"]:
        t_start_us = pd.Timestamp(task["start_time"], tz="UTC").value // 1_000
        t_end_us = pd.Timestamp(task["end_time"], tz="UTC").value // 1_000
        task_mask = (ts >= t_start_us) & (ts < t_end_us)
        if not np.any(task_mask):
            continue
        minutes = (ts[task_mask] - t_start_us) / 1_000_000.0 / 60.0
        for m, zv in zip(minutes, z[task_mask]):
            rows.append({
                "participant_id": participant_id,
                "session_index": session_index,
                "task_label": task["label"],
                "minutes_since_task_start": float(m),
                "eda_z": float(zv),
            })

    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)
