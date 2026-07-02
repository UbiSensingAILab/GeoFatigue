"""Loader joining per-minute digital-biomarker readings to GPS location and
experiment-site zone, for spatial figures (heatmap / scatter over the site
map).

Each biomarker reading is matched to its nearest-in-time GPS fix (within a
tolerance) from the participant's contextual-sensing files, then assigned the
zone polygon (resting area, flat trail, stairs, ramp) containing that fix.
Readings with no nearby GPS fix, or whose matched location falls outside all
known zones, are dropped — there is no value in plotting a physiological
sample whose location on the site is unknown.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from geofatigue.loaders.contextual import load_participant_contextual_data
from geofatigue.loaders.digital_biomarkers_csv import load_participant_biomarker_csv
from geofatigue.loaders.gps_filtering import (
    decimate_gps_trajectory,
    kalman_smooth_trajectory,
    map_match_to_centerline,
)
from geofatigue.loaders.spatial import load_all_spatial_layouts

# contextual-sensing task file name -> zone name (load_zone_polygons /
# load_zone_centerlines key). GPS files use the smartphone app's task label
# ('flat_ground'); the experiment-layout GeoJSON uses the trail's name
# ('flat_trail'). 'resting_area' has no entry: there's no dedicated GPS task
# file for it (and no centerline — it's an open area, not a path), so
# readings located there are never map-matched, only Kalman-smoothed.
_TASK_TO_ZONE = {"flat_ground": "flat_trail", "stairs": "stairs", "ramp": "ramp"}

# signal_suffix -> aggregated_per_minute value column name
DEFAULT_SIGNALS: Dict[str, str] = {
    "eda": "eda_scl_usiemens",
    "pulse-rate": "pulse_rate_bpm",
    "activity-intensity": "activity_intensity",
}

# activity_intensity is categorical (sedentary/LPA/MPA/VPA — light/moderate/
# vigorous physical activity), not numeric like the other signals. Ordinal
# codes let it go through the same numeric value_z pipeline as everything
# else (mean ordinal level per zone is standard practice for this kind of
# ordered category, much like averaging a Likert scale).
ACTIVITY_INTENSITY_LEVELS: Dict[str, int] = {"sedentary": 0, "LPA": 1, "MPA": 2, "VPA": 3}

SPATIAL_RECORD_COLUMNS = [
    "participant_id", "signal", "value", "value_z", "timestamp_us",
    "latitude", "longitude", "zone",
]


def _zscore(values: pd.Series) -> pd.Series:
    """Z-score one participant's readings of one signal against their own
    mean/std (computed from their own GPS-matched, on-site readings — not a
    dedicated pre-task rest baseline). A participant with too few matched
    readings to estimate a std (n<2, or a constant series) gets 0.0
    everywhere: no information to standardize against, and 0.0 means
    "indistinguishable from their own average" rather than introducing a
    spurious outlier.
    """
    std = values.std()
    if not std or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def load_zone_polygons(spatial_data_root: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load the experiment-site zone Polygon boundaries as one GeoDataFrame.

    Each zone GeoJSON file (resting area / flat trail / stairs / ramp)
    contains both a centerline LineString and a zone Polygon; only the
    Polygon is kept here, since boundary checks and outline drawing need a
    closed shape, not a path.

    Args:
        spatial_data_root: Directory containing the zone GeoJSON files
            (matches SPATIAL_DATA_PATH in .env).

    Returns:
        GeoDataFrame with columns {zone, geometry}, CRS EPSG:4326. `zone`
        values are the keys from load_all_spatial_layouts, e.g.
        'resting_area', 'flat_trail', 'stairs', 'ramp'.
    """
    layouts = load_all_spatial_layouts(spatial_data_root)
    zone_frames = []
    for zone_name, layout_gdf in layouts.items():
        polygons = layout_gdf[layout_gdf.geometry.geom_type == "Polygon"].copy()
        polygons["zone"] = zone_name
        zone_frames.append(polygons[["zone", "geometry"]])

    return gpd.GeoDataFrame(
        pd.concat(zone_frames, ignore_index=True), crs="EPSG:4326",
    )


def load_zone_centerlines(spatial_data_root: Union[str, Path]) -> Dict[str, LineString]:
    """Load the trail centerline LineString for each zone that has one.

    'resting_area' has no centerline (it's an open area, not a path) and is
    not included.

    Args:
        spatial_data_root: Directory containing the zone GeoJSON files
            (matches SPATIAL_DATA_PATH in .env).

    Returns:
        Dict mapping zone name ('flat_trail', 'stairs', 'ramp') to its
        LineString centerline geometry, CRS EPSG:4326.
    """
    layouts = load_all_spatial_layouts(spatial_data_root)
    centerlines = {}
    for zone_name, layout_gdf in layouts.items():
        lines = layout_gdf[layout_gdf.geometry.geom_type == "LineString"]
        if not lines.empty:
            centerlines[zone_name] = lines.geometry.iloc[0]
    return centerlines


def _participant_gps_trajectory(
    participant_id: str,
    contextual_data_root: Union[str, Path],
    zone_centerlines: Optional[Dict[str, LineString]] = None,
) -> pd.DataFrame:
    """Build one participant's cleaned GPS trajectory: decimate, Kalman-
    smooth, and (where a centerline is known for that task's zone) map-match
    each task's GPS file independently, then concatenate across tasks into
    one time-sorted DataFrame indexed by timestamp, columns
    {latitude, longitude}.

    Each task is processed on its own, before concatenation, because
    map-matching needs to know which trail centerline applies to which
    points — that's only unambiguous per task (see _TASK_TO_ZONE), not after
    tasks are merged together.

    Fixes with speed == -1.0 are dropped first: per
    geofatigue.loaders.contextual's documented schema, that's the device's
    placeholder for an invalid/unavailable GPS reading, not a real location.
    Across the dataset this is a small fraction of rows (~0.1%) but using a
    known-invalid fix to assign a physiological reading's zone would be
    wrong regardless of how rare it is.

    Args:
        zone_centerlines: From load_zone_centerlines. If None (or a task's
            zone has no entry, e.g. there is no GPS task file for resting
            area), that task's points are Kalman-smoothed but not
            map-matched.
    """
    tasks = load_participant_contextual_data(participant_id, contextual_data_root)
    if not tasks:
        raise FileNotFoundError(
            f"No contextual sensing files found for '{participant_id}' "
            f"under {contextual_data_root}"
        )
    zone_centerlines = zone_centerlines or {}

    cleaned = []
    for task_name, task_df in tasks.items():
        valid = task_df[task_df["speed"] != -1.0][["latitude", "longitude"]]
        if valid.empty:
            continue

        points = gpd.GeoDataFrame(
            valid, geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        xy = pd.DataFrame(
            {"x": points.geometry.x.to_numpy(), "y": points.geometry.y.to_numpy()},
            index=valid.index,
        )

        xy = decimate_gps_trajectory(xy)
        if xy.empty:
            continue
        xy = kalman_smooth_trajectory(xy)

        centerline = zone_centerlines.get(_TASK_TO_ZONE.get(task_name))
        if centerline is not None:
            centerline_3857 = gpd.GeoSeries([centerline], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            xy = map_match_to_centerline(xy, centerline_3857)

        cleaned_points = gpd.GeoSeries(
            gpd.points_from_xy(xy["x"], xy["y"]), crs="EPSG:3857", index=xy.index,
        ).to_crs(epsg=4326)
        cleaned.append(pd.DataFrame(
            {"latitude": cleaned_points.y, "longitude": cleaned_points.x}, index=xy.index,
        ))

    if not cleaned:
        # Files were found (tasks is non-empty) but every fix in them was
        # speed == -1.0 (invalid) — distinct from "no GPS files found" above,
        # which is the only case that should raise. Return empty rather than
        # raise: callers already handle an empty trajectory by finding zero
        # zone-matched readings, same as if GPS simply didn't cover the time
        # range a biomarker reading needed.
        return pd.DataFrame(
            columns=["latitude", "longitude"],
            index=pd.DatetimeIndex([], name="time", dtype="datetime64[us, UTC]"),
        )

    gps = pd.concat(cleaned).sort_index()
    return gps[~gps.index.duplicated(keep="first")]


def build_participant_spatial_biomarker_records(
    participant_id: str,
    phys_data_root: Union[str, Path],
    contextual_data_root: Union[str, Path],
    zone_polygons: gpd.GeoDataFrame,
    signals: Optional[Dict[str, str]] = None,
    max_gap_seconds: float = 90.0,
    zone_centerlines: Optional[Dict[str, LineString]] = None,
) -> pd.DataFrame:
    """Join one participant's per-minute biomarker readings to GPS + zone.

    For each signal in `signals`, loads the aggregated_per_minute CSV, finds
    the nearest GPS fix in time (within `max_gap_seconds`) — after that GPS
    trajectory has been decimated, Kalman-smoothed, and map-matched, see
    _participant_gps_trajectory — and looks up the zone polygon containing
    that fix. Readings with no GPS fix in tolerance, or outside all zones,
    are dropped.

    Args:
        participant_id: Participant directory name, e.g. 'p0'.
        phys_data_root: Root directory of per-minute biomarker CSVs (matches
            PHYS_DATA_PATH in .env).
        contextual_data_root: Directory of contextual-sensing GPS CSVs
            (matches CONTEXTUAL_DATA_PATH in .env).
        zone_polygons: GeoDataFrame from load_zone_polygons.
        signals: Mapping of signal_suffix -> value_column. Defaults to
            DEFAULT_SIGNALS (eda, pulse-rate).
        max_gap_seconds: Maximum allowed gap between a biomarker timestamp
            and its nearest GPS fix.
        zone_centerlines: From load_zone_centerlines. If None, GPS is
            Kalman-smoothed but not map-matched to a trail centerline.

    Returns:
        Long-format DataFrame with columns SPATIAL_RECORD_COLUMNS.

    Raises:
        FileNotFoundError: If no GPS files are found for participant_id.
    """
    signals = signals or DEFAULT_SIGNALS
    gps = _participant_gps_trajectory(participant_id, contextual_data_root, zone_centerlines)
    gps_asof = gps.reset_index().rename(columns={gps.index.name: "gps_time"})

    signal_frames = []
    for signal_suffix, value_column in signals.items():
        try:
            biomarker = load_participant_biomarker_csv(
                participant_id, phys_data_root, signal_suffix, value_column,
            )
        except FileNotFoundError:
            continue
        if biomarker.empty:
            continue

        if signal_suffix == "activity-intensity":
            biomarker[value_column] = biomarker[value_column].map(ACTIVITY_INTENSITY_LEVELS)
            biomarker = biomarker.dropna(subset=[value_column])
            if biomarker.empty:
                continue

        biomarker = biomarker.sort_values("timestamp_us").reset_index(drop=True)
        biomarker["bio_time"] = pd.to_datetime(biomarker["timestamp_us"], unit="us", utc=True)

        merged = pd.merge_asof(
            biomarker, gps_asof,
            left_on="bio_time", right_on="gps_time",
            direction="nearest", tolerance=pd.Timedelta(seconds=max_gap_seconds),
        )
        merged = merged.dropna(subset=["latitude", "longitude"])
        if merged.empty:
            continue

        merged["signal"] = signal_suffix
        merged["value"] = merged[value_column]
        signal_frames.append(merged[["signal", "value", "timestamp_us", "latitude", "longitude"]])

    if not signal_frames:
        return pd.DataFrame(columns=SPATIAL_RECORD_COLUMNS)

    combined = pd.concat(signal_frames, ignore_index=True)
    combined["participant_id"] = participant_id
    combined["value_z"] = combined.groupby("signal")["value"].transform(_zscore)

    points = gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined["longitude"], combined["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, zone_polygons, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.dropna(subset=["zone"])

    return joined[SPATIAL_RECORD_COLUMNS].reset_index(drop=True)


def build_spatial_biomarker_records(
    participant_ids: Iterable[str],
    phys_data_root: Union[str, Path],
    contextual_data_root: Union[str, Path],
    spatial_data_root: Union[str, Path],
    signals: Optional[Dict[str, str]] = None,
    max_gap_seconds: float = 90.0,
) -> pd.DataFrame:
    """Build the pooled spatial-biomarker DataFrame for many participants.

    Participants missing GPS or biomarker data entirely are skipped with a
    warning rather than raising, so one participant's missing data does not
    block the rest of the cohort.

    Args / Returns: see build_participant_spatial_biomarker_records.
    """
    zone_polygons = load_zone_polygons(spatial_data_root)
    zone_centerlines = load_zone_centerlines(spatial_data_root)

    frames = []
    for participant_id in participant_ids:
        try:
            frame = build_participant_spatial_biomarker_records(
                participant_id, phys_data_root, contextual_data_root, zone_polygons,
                signals=signals, max_gap_seconds=max_gap_seconds,
                zone_centerlines=zone_centerlines,
            )
        except FileNotFoundError as exc:
            print(f"  [WARN] Skipping {participant_id}: {exc}")
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=SPATIAL_RECORD_COLUMNS)
    return pd.concat(frames, ignore_index=True)
