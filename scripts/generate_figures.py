#!/usr/bin/env python3
"""
Generate GeoFatigue Figure 6 (elevation-physiology composite) and
Figure 7 (spatial physiology heatmap) from raw data.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --output-dir results/figures
    python scripts/generate_figures.py --participants p1 p2
    python scripts/generate_figures.py --skip-fig6   # only Figure 7
    python scripts/generate_figures.py --skip-fig7   # only Figure 6

Outputs one 300 DPI PNG per figure to --output-dir (default: figures/).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from geofatigue.figures import (
    plot_elevation_physiology_composite,
    plot_spatial_physiology_heatmap,
    save_figure,
)
from geofatigue.loaders.spatial_physiology import build_spatial_biomarker_records, load_zone_polygons


def build_spatial_biomarker_df(
    phys_data_root: Path,
    contextual_data_root: Path,
    spatial_data_root: Path,
    participants: list[str],
) -> pd.DataFrame:
    """
    Build the pooled spatial-biomarker DataFrame for Figure 7: EDA and
    pulse-rate readings joined to their nearest GPS fix and experiment-site
    zone, via
    geofatigue.loaders.spatial_physiology.build_spatial_biomarker_records.

    Returns a DataFrame with columns: participant_id, signal, value,
    timestamp_us, latitude, longitude, zone.
    """
    return build_spatial_biomarker_records(
        participants, phys_data_root, contextual_data_root, spatial_data_root,
    )


def build_elevation_physiology_profile_df(
    metadata_path: Path,
    data_root: Path,
    contextual_data_root: Path,
    spatial_data_root: Path,
    participants: list[str],
    tif_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the per-lap elevation/EDA/pulse-rate profile DataFrame for
    Figure 6: for each participant's first (chronologically earliest)
    session, and each of the three fixed-duration tasks, segments completed
    laps from their map-matched GPS trajectory
    (geofatigue.loaders.lap_trajectory.build_task_lap_trajectory),
    baseline-z-scores their per-minute EDA and pulse-rate
    (geofatigue.features.biomarker_baseline.build_session_biomarker_records_baseline_z),
    and resamples each lap's elevation/EDA-z/pulse-z onto a shared 0-100%
    round-trip position grid
    (geofatigue.features.lap_profile.build_participant_task_lap_profiles).

    When tif_dir is provided, elevation is sampled from the per-task LiDAR
    surface rasters (<task>_surface.tif, EPSG:26911) rather than from noisy
    GPS altitude, giving a survey-grade terrain profile. GPS data is still
    used for lap detection and EDA/pulse timing.

    Participants/tasks missing any required input (GPS, EDA, or pulse-rate
    data) are skipped with a warning rather than raising, so one gap does
    not block the rest of the cohort.

    Returns:
        (lap_profile_df, task_distances_m) — lap_profile_df has columns
        geofatigue.features.lap_profile.LAP_PROFILE_COLUMNS; task_distances_m
        maps task_label -> one-way centerline length in metres.
    """
    from geofatigue.loaders.digital_biomarkers_csv import load_participant_biomarker_csv
    from geofatigue.loaders.lap_trajectory import build_task_lap_trajectory
    from geofatigue.loaders.spatial_physiology import load_zone_centerlines
    from geofatigue.features.biomarker_baseline import build_session_biomarker_records_baseline_z
    from geofatigue.features.lap_profile import LAP_PROFILE_COLUMNS, build_participant_task_lap_profiles

    session_json = metadata_path / "session_metadata.json"
    with open(session_json) as f:
        session_meta = json.load(f)

    centerlines_4326 = load_zone_centerlines(spatial_data_root)
    centerlines_3857 = {
        zone: gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        for zone, line in centerlines_4326.items()
    }
    # One-way centerline lengths (metres, EPSG:3857) for distance-proportional
    # x-axis block widths in Figure 6.
    task_distances_m = {
        task: float(centerlines_3857[task].length)
        for task in ["flat_trail", "stairs", "ramp"]
        if task in centerlines_3857
    }

    # Pre-sample LiDAR elevation profiles once per task (same terrain for all
    # participants). Falls back to per-lap GPS altitude when tif_dir is None.
    lidar_profiles: dict[str, tuple] = {}
    if tif_dir is not None:
        from geofatigue.loaders.tif_elevation import sample_centerline_elevation_profile
        for task_label in ["flat_trail", "stairs", "ramp"]:
            cl = centerlines_4326.get(task_label)
            if cl is not None:
                try:
                    lidar_profiles[task_label] = sample_centerline_elevation_profile(
                        tif_dir, task_label, cl,
                    )
                except FileNotFoundError as exc:
                    print(f"  [WARN] LiDAR TIF missing for {task_label}: {exc}; falling back to GPS altitude")
        if lidar_profiles:
            print(f"  LiDAR elevation loaded for: {', '.join(lidar_profiles)}")

    frames = []

    task_labels = ["flat_trail", "stairs", "ramp"]
    total_candidates = len(participants)
    skip_counts = {"no_session": 0, "no_biomarker_files": 0, "baseline_error": 0}
    task_included: dict[str, set] = {t: set() for t in task_labels}
    task_missing_gps = {t: 0 for t in task_labels}
    task_zero_laps = {t: 0 for t in task_labels}
    task_no_task_or_centerline = {t: 0 for t in task_labels}

    for pid in participants:
        sessions = session_meta.get(pid, [])
        if not sessions:
            skip_counts["no_session"] += 1
            continue
        session = min(sessions, key=lambda s: s["start_time"])

        try:
            eda_signal_df = load_participant_biomarker_csv(pid, data_root, "eda", "eda_scl_usiemens")
            pulse_signal_df = load_participant_biomarker_csv(pid, data_root, "pulse-rate", "pulse_rate_bpm")
        except FileNotFoundError as exc:
            skip_counts["no_biomarker_files"] += 1
            print(f"  [WARN] Skipping {pid}: {exc}")
            continue

        try:
            # EDA (skin-conductance level) is right-skewed and heteroscedastic
            # across rest vs. activity; log-transforming before baseline
            # z-scoring is the standard remedy (see
            # build_session_biomarker_records_baseline_z's log_transform
            # docstring). Pulse rate (bpm) has no such skew, so it doesn't.
            eda_z_df = build_session_biomarker_records_baseline_z(
                pid, 0, session, eda_signal_df, "eda_scl_usiemens", log_transform=True,
            )
            pulse_z_df = build_session_biomarker_records_baseline_z(pid, 0, session, pulse_signal_df, "pulse_rate_bpm")
        except ValueError as exc:
            skip_counts["baseline_error"] += 1
            print(f"  [WARN] Skipping {pid}: {exc}")
            continue

        for task_label in task_labels:
            task = next((t for t in session["tasks"] if t["label"] == task_label), None)
            centerline = centerlines_3857.get(task_label)
            if task is None or centerline is None:
                task_no_task_or_centerline[task_label] += 1
                continue

            try:
                gps_trajectory = build_task_lap_trajectory(pid, task_label, contextual_data_root, centerline)
            except FileNotFoundError as exc:
                task_missing_gps[task_label] += 1
                print(f"  [WARN] Skipping {pid}/{task_label}: {exc}")
                continue

            eda_task_z = eda_z_df[eda_z_df["task_label"] == task_label]
            pulse_task_z = pulse_z_df[pulse_z_df["task_label"] == task_label]
            task_start_time = pd.Timestamp(task["start_time"], tz="UTC")

            frame = build_participant_task_lap_profiles(
                pid, 0, task_label, gps_trajectory, centerline,
                eda_task_z, pulse_task_z, task_start_time=task_start_time,
                lidar_elevation_profile=lidar_profiles.get(task_label),
            )
            if not frame.empty:
                frames.append(frame)
                task_included[task_label].add(pid)
            else:
                task_zero_laps[task_label] += 1

    print(f"\n  Elevation-physiology attrition: {total_candidates} candidate participants")
    print(f"    Skipped (no session metadata): {skip_counts['no_session']}")
    print(f"    Skipped (no EDA/pulse-rate CSV found): {skip_counts['no_biomarker_files']}")
    print(f"    Skipped (baseline window/zero-variance error): {skip_counts['baseline_error']}")
    for task_label in task_labels:
        print(
            f"    {task_label}: included={len(task_included[task_label])}, "
            f"missing GPS file={task_missing_gps[task_label]}, "
            f"zero completed laps={task_zero_laps[task_label]}, "
            f"no task entry/centerline={task_no_task_or_centerline[task_label]}"
        )

    if not frames:
        return pd.DataFrame(columns=LAP_PROFILE_COLUMNS), task_distances_m
    return pd.concat(frames, ignore_index=True), task_distances_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GeoFatigue Figure 6 and Figure 7.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root directory of raw AVRO data (default: $DATA_ROOT from .env)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Metadata directory (default: $METADATA_PATH from .env)",
    )
    parser.add_argument(
        "--contextual-dir",
        type=Path,
        default=None,
        help="Contextual sensing (GPS) data directory (default: $CONTEXTUAL_DATA_PATH from .env)",
    )
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=None,
        help="Experiment layout (zone GeoJSON) directory (default: $SPATIAL_DATA_PATH from .env)",
    )
    parser.add_argument(
        "--tif-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <task>_surface.tif LiDAR rasters for Figure 6 elevation "
            "(default: $TIF_DIR from .env; falls back to GPS altitude if unset)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for PNG files (default: figures/)",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=None,
        help="Subset of participant IDs (default: all participants in session_metadata.json)",
    )
    parser.add_argument(
        "--skip-fig6",
        action="store_true",
        help="Skip Figure 6 (elevation-physiology composite)",
    )
    parser.add_argument(
        "--skip-fig7",
        action="store_true",
        help="Skip Figure 7 (spatial physiology heatmap)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    data_root = args.data_dir or Path(os.environ["DATA_ROOT"])
    metadata_path = args.metadata_dir or Path(os.environ["METADATA_PATH"])
    contextual_data_root = args.contextual_dir or Path(os.environ["CONTEXTUAL_DATA_PATH"])
    spatial_data_root = args.spatial_dir or Path(os.environ["SPATIAL_DATA_PATH"])
    tif_dir = args.tif_dir or (Path(os.environ["TIF_DIR"]) if os.environ.get("TIF_DIR") else None)
    output_dir = args.output_dir or (Path(os.environ["OUTPUT_DIR"]) if os.environ.get("OUTPUT_DIR") else None)

    print(f"Output directory : {output_dir}")
    print(f"Metadata         : {metadata_path}")

    with open(metadata_path / "session_metadata.json") as f:
        session_meta = json.load(f)
    participants = args.participants or list(session_meta.keys())

    if not args.skip_fig7:
        print("\nBuilding spatial biomarker DataFrame (loads aggregated_per_minute CSVs + GPS)...")
        spatial_df = build_spatial_biomarker_df(
            phys_data_root=data_root,
            contextual_data_root=contextual_data_root,
            spatial_data_root=spatial_data_root,
            participants=participants,
        )
        if spatial_df.empty:
            print("  [WARN] No spatially-located physiological data found — skipping Figure 7")
        else:
            print(f"  {len(spatial_df)} rows across {spatial_df['participant_id'].nunique()} participants")
            for signal, count in spatial_df["signal"].value_counts().items():
                print(f"    {signal}: {count}")

            zone_polygons = load_zone_polygons(spatial_data_root)

            print("\nGenerating Figure 7 (spatial physiology heatmap)...")
            path = save_figure(
                plot_spatial_physiology_heatmap(spatial_df, zone_polygons),
                "fig7_spatial_physiology_heatmap", output_dir,
            )
            print(f"  Saved → {path}")
    else:
        print("\nSkipping Figure 7 (--skip-fig7 set)")

    if not args.skip_fig6:
        from geofatigue.features.lap_profile import build_caption_stats

        print("\nBuilding elevation-physiology lap profile DataFrame (Figure 6)...")
        lap_profile_df, task_distances_m = build_elevation_physiology_profile_df(
            metadata_path=metadata_path, data_root=data_root,
            contextual_data_root=contextual_data_root, spatial_data_root=spatial_data_root,
            participants=participants,
            tif_dir=tif_dir,
        )
        if lap_profile_df.empty:
            print("  [WARN] No lap data found — skipping Figure 6")
        else:
            print(f"  {len(lap_profile_df)} rows across {lap_profile_df['participant_id'].nunique()} participants")
            caption_stats = build_caption_stats(lap_profile_df)
            for task, s in caption_stats.items():
                print(
                    f"    {task}: n={s['n_participants']}, laps={s['n_laps']} "
                    f"({s['laps_per_participant_mean']:.1f} ± {s['laps_per_participant_sd']:.1f} per participant)"
                )
            print("\nGenerating Figure 6 (elevation-physiology composite)...")
            path = save_figure(
                plot_elevation_physiology_composite(lap_profile_df, task_distances_m=task_distances_m),
                "fig6_elevation_physiology_composite", output_dir,
            )
            print(f"  Saved → {path}")
    else:
        print("\nSkipping Figure 6 (--skip-fig6 set)")

    print(f"\nDone. Figures written to {output_dir}/")


if __name__ == "__main__":
    main()
