"""Noise reduction for pedestrian GPS traces: decimation, Kalman smoothing,
and map-matching to a known trail centerline.

Three independent, composable steps, meant to be applied in this order:

  1. decimate_gps_trajectory — average raw fixes into 1-second buckets.
     At ~100 Hz, a walking pedestrian (~1 m/s) moves ~1 cm between
     consecutive samples — far below GPS accuracy (~5-10 m) — so most of
     those samples are repeated noisy estimates of the same true position,
     not new information. Averaging is itself a basic noise-reduction step
     and makes the next step computationally trivial (~100x fewer points).
  2. kalman_smooth_trajectory — a constant-velocity Kalman filter applied
     independently to x and y (planar meters), pulling each fix toward a
     locally-linear prediction of the trajectory.
  3. map_match_to_centerline — snap each smoothed point onto the nearest
     point of a known trail centerline, when one is available for the zone
     a participant was walking in. This uses domain knowledge this
     dataset's GPS loader doesn't otherwise have: participants walked
     continuously along a fixed path (flat trail / stairs / ramp), so any
     remaining cross-track offset is GPS noise, not real lateral movement.

All functions operate on (and return) a DataFrame with a DatetimeIndex and
columns {x, y} in planar meters (e.g. Web Mercator, EPSG:3857) — callers
handle CRS conversion to/from lon/lat; see
geofatigue.loaders.spatial_physiology._participant_gps_trajectory.
"""

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def decimate_gps_trajectory(df: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """Average {x, y} into one fix per `freq` time bucket.

    Empty buckets (no raw fixes in that second) are dropped rather than
    interpolated, so the result can have irregular spacing across gaps —
    kalman_smooth_trajectory handles variable time steps explicitly.
    """
    return df.resample(freq).mean().dropna()


def kalman_smooth_trajectory(
    df: pd.DataFrame,
    process_noise_std: float = 0.3,
    measurement_noise_std: float = 5.0,
) -> pd.DataFrame:
    """Constant-velocity Kalman filter, applied independently to x and y.

    Each axis is a standard 2-state [position, velocity] linear system with
    scalar position measurements. Time steps are computed from the actual
    index timestamps (not assumed constant), so gaps from decimation just
    widen that step's process-noise covariance rather than breaking the
    filter.

    Args:
        df: DataFrame with a DatetimeIndex and columns {x, y} (meters).
        process_noise_std: Assumed 1-sigma acceleration noise (m/s^2) — how
            much a walking pedestrian's velocity can plausibly change
            between samples. Not fit to this dataset; a reasonable default
            for pedestrian motion, tunable.
        measurement_noise_std: Assumed 1-sigma GPS measurement error (m).
            ~5 m is typical for unassisted consumer smartphone GPS.

    Returns:
        DataFrame with the same index, columns {x, y} replaced by their
        smoothed values. Inputs with fewer than 2 rows are returned
        unchanged (nothing to smooth against).
    """
    if len(df) < 2:
        return df.copy()

    t_seconds = df.index.values.astype("datetime64[ns]").astype("int64") / 1e9
    dt = np.diff(t_seconds, prepend=t_seconds[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.clip(dt, 1e-3, None)

    result = df.copy()
    q = process_noise_std ** 2
    r = measurement_noise_std ** 2

    for axis in ("x", "y"):
        z = df[axis].to_numpy()
        state = np.array([z[0], 0.0])  # [position, velocity]
        cov = np.eye(2) * r
        smoothed = np.empty(len(z))
        smoothed[0] = state[0]

        for i in range(1, len(z)):
            step = dt[i]
            F = np.array([[1.0, step], [0.0, 1.0]])
            Q = q * np.array([
                [step ** 3 / 3, step ** 2 / 2],
                [step ** 2 / 2, step],
            ])

            state = F @ state
            cov = F @ cov @ F.T + Q

            innovation = z[i] - state[0]
            innovation_var = cov[0, 0] + r
            gain = cov[:, 0] / innovation_var
            state = state + gain * innovation
            cov = cov - np.outer(gain, cov[0, :])

            smoothed[i] = state[0]

        result[axis] = smoothed

    return result


def map_match_to_centerline(df: pd.DataFrame, centerline: LineString) -> pd.DataFrame:
    """Snap every (x, y) point onto the nearest point on `centerline`.

    Uses shapely's project/interpolate (arc-length projection onto the
    line), the standard "snap to path" operation — appropriate here because
    participants are known to have walked along this exact centerline, so
    perpendicular (cross-track) offset is treated entirely as GPS noise.

    Args:
        df: DataFrame with columns {x, y} in the same planar CRS as
            `centerline`.
        centerline: The trail's centerline geometry.

    Returns:
        DataFrame with the same index, columns {x, y} replaced by their
        on-line projected values.
    """
    arc_lengths = [centerline.project(Point(x, y)) for x, y in zip(df["x"], df["y"])]
    snapped = [centerline.interpolate(d) for d in arc_lengths]

    result = df.copy()
    result["x"] = [p.x for p in snapped]
    result["y"] = [p.y for p in snapped]
    return result
