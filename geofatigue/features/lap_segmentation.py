"""Pure functions for turning a participant's raw position-vs-time trace
along a task's centerline into well-defined completed laps and a 0-100%
round-trip position axis within each lap.

A "lap" is a closed there-and-back trip: from the task's fixed lap-start
point (an extremity of the centerline, see TASK_START_DIRECTION) away to the
turnaround point and back. 
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

# Each task's lap-start point is the centerline endpoint at this geographic
# extreme (confirmed against the real site layout -- see the design spec).
TASK_START_DIRECTION = {
    "flat_trail": "west",
    "stairs": "east",
    "ramp": "north",
}

_DIRECTION_AXIS = {"west": 0, "east": 0, "north": 1, "south": 1}
_DIRECTION_PICKS_MIN = {"west": True, "east": False, "north": False, "south": True}


def pick_extreme_endpoint(centerline: LineString, direction: str) -> Point:
    """Return whichever of the centerline's two endpoints is furthest in
    `direction` ('west'/'east' compare x/longitude; 'north'/'south' compare
    y/latitude -- works for any planar CRS where larger x = east and larger
    y = north, as well as raw WGS84 lon/lat).

    Compares actual coordinates rather than assuming a fixed endpoint index,
    since a centerline's point order in its source GeoJSON is incidental.
    """
    if direction not in _DIRECTION_AXIS:
        raise ValueError(f"Unknown direction: {direction!r}")
    axis = _DIRECTION_AXIS[direction]
    wants_min = _DIRECTION_PICKS_MIN[direction]

    coords = list(centerline.coords)
    first, last = Point(coords[0]), Point(coords[-1])
    first_val = first.coords[0][axis]
    last_val = last.coords[0][axis]

    if wants_min:
        return first if first_val <= last_val else last
    return first if first_val >= last_val else last


def distance_from_start(
    centerline: LineString, start_point: Point, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Distance traveled-along-centerline between each (x, y) and the lap
    start point -- i.e. abs(arc-length(point) - arc-length(start)). Zero
    exactly at the start point, growing as a participant moves away along
    the path in either direction (the lap-start point is always one
    extreme end of the centerline, so this is never ambiguous)."""
    start_arc = centerline.project(start_point)
    arcs = np.array([centerline.project(Point(x, y)) for x, y in zip(xs, ys)])
    return np.abs(arcs - start_arc)


def find_completed_laps(
    timestamps: pd.DatetimeIndex,
    distance_from_start_m: np.ndarray,
    tolerance_m: float = 2.0,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Partition a task's trace into completed (start -> away -> back to
    start) laps, using `tolerance_m` as how close counts as "at the start
    point".

    Consecutive samples within tolerance are collapsed into a single
    boundary event (the first sample of each such run) so GPS noise
    wobbling across the tolerance edge doesn't fragment one visit into many.
    Returns one (lap_start, lap_end) pair per pair of consecutive boundary
    events -- the trailing segment after the last boundary (truncated by the
    task's fixed duration) is never a completed lap and is dropped.
    """
    if len(timestamps) == 0:
        return []

    at_start = distance_from_start_m <= tolerance_m
    boundary_times = [
        timestamps[i] for i in range(len(at_start))
        if at_start[i] and (i == 0 or not at_start[i - 1])
    ]
    return list(zip(boundary_times[:-1], boundary_times[1:]))


def lap_round_trip_pct(arc_length_m: np.ndarray) -> np.ndarray:
    """0-100% position within one lap: cumulative along-centerline distance
    traveled so far, as a fraction of the lap's total round-trip distance.
    0% = the lap start, 100% = back at the start; the turnaround point (the
    far end of the path) falls at ~50% regardless of how long each leg
    actually took, since this tracks distance traveled, not elapsed time.

    `arc_length_m` must already be one lap's raw along-centerline positions,
    in time order (e.g. centerline.project() output, not distance-from-start
    -- direction reversals are what need to show up as the cumulative-sum
    going up on both legs).
    """
    if len(arc_length_m) < 2:
        return np.zeros_like(arc_length_m, dtype=float)
    steps = np.abs(np.diff(arc_length_m))
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    if total == 0:
        return np.zeros_like(cumulative)
    return cumulative / total * 100.0
