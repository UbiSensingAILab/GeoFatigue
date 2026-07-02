"""Per-task, altitude-retaining, map-matched GPS trajectory -- the
geometric basis for lap segmentation in the elevation-physiology composite
figure (fig6). Reuses the decimate/Kalman/map-match pipeline from
geofatigue.loaders.gps_filtering, the same one
geofatigue.loaders.spatial_physiology._participant_gps_trajectory uses, but
keeps altitude (dropped there) and adds each point's along-centerline arc
length, needed to fold a lap's outbound+inbound legs into one 0-100% axis.

Unlike x/y, raw GPS altitude gets no smoothing anywhere in
gps_filtering.py's pipeline (its Kalman filter only touches x/y) -- but real
fixes show isolated multi-metre-to-tens-of-metres altitude spikes (poor sky
view / multipath). Left unfiltered, these don't average out across
laps/participants the way independent per-fix noise would, because every
lap starts at the same physical point, so a spike tied to that location
recurs at the same round-trip position every time. Despiked here with the
same Hampel filter geofatigue.features.eda.py already uses for raw EDA.
"""
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from geofatigue.filters.signal_filter import hampel
from geofatigue.loaders.contextual import load_participant_contextual_data
from geofatigue.loaders.gps_filtering import (
    decimate_gps_trajectory,
    kalman_smooth_trajectory,
    map_match_to_centerline,
)

# Decimated altitude is ~1 sample/second; a 5-second window catches isolated
# spike fixes without smoothing away genuine elevation change over a task's
# ~10-40 s legs (flat trail/stairs/ramp are all walked at ~1 m/s).
ALTITUDE_HAMPEL_WINDOW = 5

# session_metadata.json / centerline-file task naming -> contextual-sensing
# file's own task label (mirrors geofatigue.loaders.spatial_physiology's
# _TASK_TO_ZONE, inverted).
TASK_FILENAME_KEYS = {"flat_trail": "flat_ground", "stairs": "stairs", "ramp": "ramp"}

_TRAJECTORY_COLUMNS = ["x", "y", "altitude", "arc_length_m"]


def _empty_trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        columns=_TRAJECTORY_COLUMNS,
        index=pd.DatetimeIndex([], name="time", dtype="datetime64[us, UTC]"),
    )


def build_task_lap_trajectory(
    participant_id: str,
    task_label: str,
    contextual_data_root: Union[str, Path],
    centerline_3857: LineString,
) -> pd.DataFrame:
    """One participant's one-task GPS trace, cleaned and map-matched, with
    altitude retained and each point's arc length along `centerline_3857`
    computed.

    Args:
        task_label: One of 'flat_trail', 'stairs', 'ramp' (the
            session_metadata.json / centerline-file naming) -- translated to
            the contextual-sensing file's own task label via
            TASK_FILENAME_KEYS.
        centerline_3857: The task's centerline, already reprojected to
            EPSG:3857 (planar meters) -- same CRS this function's output
            uses for x/y and arc length.

    Returns:
        DataFrame indexed by timestamp (UTC), columns {x, y, altitude,
        arc_length_m}, sorted by time. Empty (same columns, no rows) if
        every fix is invalid (speed == -1.0).

    Raises:
        FileNotFoundError: If no contextual sensing file exists at all for
            this participant/task.
    """
    filename_key = TASK_FILENAME_KEYS[task_label]
    tasks = load_participant_contextual_data(
        participant_id, contextual_data_root, tasks=[filename_key],
    )
    if filename_key not in tasks:
        raise FileNotFoundError(
            f"No contextual sensing file found for '{participant_id}' task '{task_label}' "
            f"under {contextual_data_root}"
        )

    raw = tasks[filename_key]
    valid = raw[raw["speed"] != -1.0][["latitude", "longitude", "altitude"]]
    if valid.empty:
        return _empty_trajectory()

    points = gpd.GeoDataFrame(
        valid, geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    xy = pd.DataFrame(
        {
            "x": points.geometry.x.to_numpy(),
            "y": points.geometry.y.to_numpy(),
            "altitude": valid["altitude"].to_numpy(),
        },
        index=valid.index,
    )

    xy = decimate_gps_trajectory(xy)
    if xy.empty:
        return _empty_trajectory()
    xy["altitude"] = hampel(xy["altitude"].to_numpy(), window_size=ALTITUDE_HAMPEL_WINDOW)
    xy = kalman_smooth_trajectory(xy)
    xy = map_match_to_centerline(xy, centerline_3857)

    xy["arc_length_m"] = [
        centerline_3857.project(Point(x, y)) for x, y in zip(xy["x"], xy["y"])
    ]
    return xy.sort_index()
