"""LiDAR-derived elevation sampling along task centerlines.

Reads the per-task 5 cm surface rasters (produced from the registered TLS
point cloud, EPSG:26911 / NAD83 UTM Zone 11N) and returns a reference
elevation profile for each task's round-trip lap position axis (0–100%).

This replaces the noisy (~5–15 m vertical error) GPS altitude previously
used in geofatigue.features.lap_profile with survey-grade terrain elevation.
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import median_filter as _ndimage_median
from shapely.geometry import LineString, Point

from geofatigue.features.lap_segmentation import TASK_START_DIRECTION, pick_extreme_endpoint
from geofatigue.filters.signal_filter import hampel

_TIF_SUFFIX = "_surface.tif"

# Stairs DSM pre-processing: a 2D spatial median filter of this size (pixels)
# is applied to the raster before sampling.  At 5 cm resolution a 9×9 kernel
# (45 cm × 45 cm) removes features narrower than ~20 cm — handrail pipes
# (4–10 cm), handrail cap plates (10–15 cm), step nosings (2–3 cm), and
# bollards/posts — while preserving stair treads (~25–30 cm deep = 5–6 pixels).
# A 5×5 kernel is insufficient for handrails whose DSM footprint spans 3 pixels
# (15 cm): 3×5 = 15 of 25 pixels are elevated, so the median stays on the
# handrail.  With 9×9: 3×9 = 27 of 81 pixels → 33% → median snaps to tread.
# No lateral offset is used: a fixed perpendicular shift can land all samples
# on the structural side wall at the stair base, causing elevation errors
# > 50 cm even though the underlying LiDAR has 2 mm point accuracy.
_STAIRS_DSM_FILTER_SIZE: int = 9


def sample_centerline_elevation_profile(
    tif_dir: Path,
    task_label: str,
    centerline_4326: LineString,
    n_grid_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample LiDAR surface-raster elevation along a task's round-trip path.

    Walks the centerline from its lap-start endpoint (per TASK_START_DIRECTION)
    to the far end (outbound leg) and back (return leg), sampling n_grid_points
    equally-spaced points. The resulting profile matches the 0–100% round-trip
    position axis used in build_participant_task_lap_profiles: 0% and 100% are
    at the lap-start point, ~50% is at the turnaround (far end).

    For the stairs task the DSM is pre-filtered with a 2D spatial median kernel
    (_STAIRS_DSM_FILTER_SIZE × _STAIRS_DSM_FILTER_SIZE pixels) before sampling.
    This removes sub-tread structural artifacts (handrail pipes, step nosings,
    bollards) which appear as narrow elevated ridges in the DSM.  The median
    filter preserves the broad flat surfaces of each stair tread so any
    centerline sample position reliably returns the walkable surface elevation.

    Args:
        tif_dir: Directory containing <task_label>_surface.tif files.
        task_label: One of 'flat_trail', 'stairs', 'ramp'.
        centerline_4326: Task centerline in WGS84 (EPSG:4326), as returned by
            geofatigue.loaders.spatial_physiology.load_zone_centerlines.
        n_grid_points: Must match the n_grid_points used in
            build_participant_task_lap_profiles (default 100).

    Returns:
        (position_pct, elevation_m) — two arrays of length n_grid_points.
        position_pct == np.linspace(0, 100, n_grid_points).
        Nodata pixels are filled by linear interpolation from neighbours.

    Raises:
        FileNotFoundError: If <tif_dir>/<task_label>_surface.tif does not exist.
    """
    tif_path = tif_dir / f"{task_label}{_TIF_SUFFIX}"
    if not tif_path.exists():
        raise FileNotFoundError(f"LiDAR TIF not found: {tif_path}")

    # Orient the centerline so coords[0] is the lap-start endpoint.
    start_pt = pick_extreme_endpoint(centerline_4326, TASK_START_DIRECTION[task_label])
    coords = list(centerline_4326.coords)
    if Point(coords[0]).distance(start_pt) > 1e-9:
        coords = list(reversed(coords))

    # Build round-trip LineString: outbound then return, without duplicating
    # the turnaround point where the two legs meet.
    round_trip = LineString(coords + list(reversed(coords))[1:])

    # Sample n_grid_points equally-spaced points along the round trip.
    fracs = np.linspace(0.0, 1.0, n_grid_points)
    sample_pts = [round_trip.interpolate(f, normalized=True) for f in fracs]
    xs_4326 = [p.x for p in sample_pts]
    ys_4326 = [p.y for p in sample_pts]

    # Reproject to the TIF's CRS (EPSG:26911, NAD83 / UTM Zone 11N).
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26911", always_xy=True)
    xs_26911, ys_26911 = transformer.transform(xs_4326, ys_4326)

    with rasterio.open(tif_path) as src:
        nodata = src.nodata

        if task_label == "stairs":
            # Read the full raster, apply 2D spatial median filter, then sample.
            data = src.read(1).astype(float)

            # Track nodata locations; fill with local median so the filter
            # doesn't propagate the sentinel value into valid neighbours.
            nodata_mask = np.zeros(data.shape, dtype=bool)
            if nodata is not None:
                nodata_mask = data == nodata
                if nodata_mask.any() and not nodata_mask.all():
                    data[nodata_mask] = float(np.nanmedian(data[~nodata_mask]))

            filtered = _ndimage_median(data, size=_STAIRS_DSM_FILTER_SIZE)

            # Convert reprojected sample coordinates to raster row/col indices.
            rows, cols = zip(*[src.index(x, y) for x, y in zip(xs_26911, ys_26911)])
            rows = np.clip(np.array(rows), 0, src.height - 1)
            cols = np.clip(np.array(cols), 0, src.width - 1)
            elevation_m = filtered[rows, cols].astype(float)

            # Restore nodata where the sampled pixels were invalid.
            if nodata is not None and nodata_mask.any():
                bad = nodata_mask[rows, cols]
                if bad.any() and not bad.all():
                    good_idx = np.where(~bad)[0]
                    elevation_m = np.interp(
                        np.arange(len(elevation_m)), good_idx, elevation_m[good_idx]
                    )
                elif bad.all():
                    elevation_m[:] = np.nan

        else:
            elevation_m = np.array(
                [v[0] for v in src.sample(zip(xs_26911, ys_26911))], dtype=float
            )
            if nodata is not None:
                bad = elevation_m == nodata
                if bad.any() and not bad.all():
                    good_idx = np.where(~bad)[0]
                    elevation_m = np.interp(
                        np.arange(len(elevation_m)), good_idx, elevation_m[good_idx]
                    )
                elif bad.all():
                    elevation_m[:] = np.nan

    # Non-stairs: Hampel filter removes isolated GPS/raster outliers.
    if task_label != "stairs" and not np.any(np.isnan(elevation_m)):
        elevation_m = hampel(elevation_m, window_size=5)

    # Stairs: tent ceiling at the turnaround zone (top landing).
    # The maximum legitimate elevation in the landing zone is bounded by the
    # highest clean stair step just outside the window (the last step on the
    # ascending side and the first step on the descending side, both measured
    # clear of the landing structure) plus a 10 cm tolerance (< ½ riser).
    # Any DSM value inside the window that exceeds this ceiling is a
    # structural artefact — handrail end-cap, gate post, landing fence post
    # — and is clipped.  This handles symmetric artefacts (same physical
    # structure sampled on both ascending and descending passes) that the
    # mirror approach cannot remove.  The flat approach sections (positions
    # 0–24% and 76–100%) are excluded from the reference windows by design.
    if task_label == "stairs" and not np.any(np.isnan(elevation_m)):
        n = len(elevation_m)
        mid = n // 2
        # Window is mid±9 so the spike edges (positions 41 and 58) fall
        # inside the correction zone rather than inside the reference window.
        lo = max(5, mid - 9)
        hi = min(n - 5, mid + 10)
        n_ref = 5

        E_asc  = np.max(elevation_m[lo - n_ref : lo])
        E_desc = np.max(elevation_m[hi : hi + n_ref])
        ceiling = max(E_asc, E_desc) + 0.07
        elevation_m[lo:hi] = np.minimum(elevation_m[lo:hi], ceiling)

    # Stairs: anchor start/end to flat-ground elevation at the stair base.
    # Position 0% and 100% are the same physical location (the stair entrance
    # connecting to the flat trail).  Even after 2D filtering, the centerline
    # endpoint pixel can straddle a step riser; the minimum of the nearest
    # samples gives the flat-ground approach elevation.
    if task_label == "stairs" and not np.any(np.isnan(elevation_m)):
        anchor_n = 8
        elevation_m[0] = elevation_m[:anchor_n].min()
        elevation_m[-1] = elevation_m[-anchor_n:].min()

    return np.linspace(0.0, 100.0, n_grid_points), elevation_m
