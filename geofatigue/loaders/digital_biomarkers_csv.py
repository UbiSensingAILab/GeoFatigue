"""Generic loader for Empatica's per-minute aggregated digital-biomarker CSV exports.

Every aggregated_per_minute/*_<signal>.csv file shares the same schema —
timestamp_unix, timestamp_iso, participant_full_id, <value_column>,
missing_value_reason — for whichever signal the suffix names (eda,
pulse-rate, prv, temperature, ...). This loader works for any of them given
the file suffix and value column name, so adding a new signal needs no new
loader code.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_participant_biomarker_csv(
    participant_id: str,
    data_root: Union[str, Path],
    signal_suffix: str,
    value_column: str,
) -> pd.DataFrame:
    """Load and concatenate all aggregated_per_minute `<signal_suffix>` CSVs
    for one participant.

    Scans `data_root` recursively for `*_<signal_suffix>.csv` files living in
    an `aggregated_per_minute` directory, under a participant folder named
    `participant_id` or `participant_id-<hardware_serial>` (one date
    sub-directory per session day, e.g. `data_root/2025-06-03/p0-3YK9R1L11Y/...`).

    Rows with a `missing_value_reason` (sensor not worn/recording) are dropped.

    Args:
        participant_id: Participant directory name, e.g. 'p0'.
        data_root: Root directory containing one date sub-directory per
            recording day (matches DATA_ROOT / PHYS_DATA_PATH in .env).
        signal_suffix: File-name signal tag, e.g. 'eda', 'pulse-rate', 'prv',
            'temperature'.
        value_column: Name of the signal's value column in the CSV, e.g.
            'eda_scl_usiemens', 'pulse_rate_bpm'.

    Returns:
        DataFrame sorted by timestamp with columns: timestamp_us, <value_column>.

    Raises:
        FileNotFoundError: If no matching CSV files are found.
    """
    data_root = Path(data_root)
    csv_files = sorted(
        f for f in data_root.rglob(f"*_{signal_suffix}.csv")
        if f.parent.name == "aggregated_per_minute"
        and any(
            part == participant_id or part.startswith(f"{participant_id}-")
            for part in f.relative_to(data_root).parts
        )
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No aggregated_per_minute '{signal_suffix}' CSVs found for "
            f"'{participant_id}' under {data_root}"
        )

    frames = [pd.read_csv(f, usecols=["timestamp_unix", value_column]) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True).dropna(subset=[value_column])
    combined["timestamp_us"] = combined["timestamp_unix"] * 1_000
    combined = combined.drop(columns=["timestamp_unix"])
    return combined.sort_values("timestamp_us").reset_index(drop=True)
