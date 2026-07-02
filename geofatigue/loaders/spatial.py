from __future__ import annotations
"""
Loader for spatial data (GeoJSON experiment layout).
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
try:
    import geopandas as gpd
    from shapely.geometry import shape, Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def load_spatial_layout(
    file_path: Union[str, Path],
    as_geopandas: bool = True
) -> Union[gpd.GeoDataFrame, Dict]:
    """
    Load spatial layout from GeoJSON file.

    The GeoJSON files define the spatial boundaries of different zones in the
    experiment site, including the resting area, flat trail, stairs, and ramp.

    Parameters
    ----------
    file_path : str or Path
        Path to the GeoJSON file (e.g., 'resting area.geojson', 'flat trail.geojson')
    as_geopandas : bool, optional
        If True and geopandas is available, return as GeoDataFrame.
        If False or geopandas not available, return as dict. Default: True

    Returns
    -------
    geopandas.GeoDataFrame or dict
        Spatial features with geometry and properties.
        If GeoDataFrame, includes a 'geometry' column with Shapely geometries.

    Example
    -------
    >>> # Load resting area
    >>> resting = load_spatial_layout('path/to/resting area.geojson')
    >>> print(f"Features: {len(resting)}")
    >>> print(f"CRS: {resting.crs}")
    >>> print(f"Total area: {resting.geometry.area.sum()} square degrees")
    >>>
    >>> # Load flat trail
    >>> trail = load_spatial_layout('path/to/flat trail.geojson')
    >>> print(f"Geometry types: {trail.geometry.geom_type.unique()}")
    >>>
    >>> # Check if a point is within the resting area
    >>> point = Point(-114.13177, 51.07956)
    >>> is_in_rest = resting.geometry.contains(point).any()
    >>> print(f"Point is in resting area: {is_in_rest}")

    Notes
    -----
    - Coordinate Reference System: WGS84 (EPSG:4326)
    - Geometry types vary by file:
        * resting area: Polygon
        * flat trail: LineString (centerline) + Polygon (zone)
        * stairs: LineString (path) + Polygon (zone)
        * ramp: LineString (path) + Polygon (zone)
    - Properties include 'name' field identifying the feature
    - Coordinates are in [longitude, latitude] order (GeoJSON standard)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        geojson_data = json.load(f)

    if as_geopandas and GEOPANDAS_AVAILABLE:
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
        gdf = gdf.set_crs("EPSG:4326")
        return gdf
    else:
        return geojson_data


def load_all_spatial_layouts(
    data_directory: Union[str, Path],
    as_geopandas: bool = True
) -> Dict[str, Union[gpd.GeoDataFrame, Dict]]:
    """
    Load all spatial layout files from a directory.

    Parameters
    ----------
    data_directory : str or Path
        Directory containing GeoJSON layout files
    as_geopandas : bool, optional
        If True and geopandas is available, return as GeoDataFrames. Default: True

    Returns
    -------
    dict
        Dictionary mapping zone names to spatial data:
        {
            'resting_area': GeoDataFrame or dict,
            'flat_trail': GeoDataFrame or dict,
            'stairs': GeoDataFrame or dict,
            'ramp': GeoDataFrame or dict
        }

    Example
    -------
    >>> layouts = load_all_spatial_layouts('path/to/layout_directory/')
    >>> for zone_name, gdf in layouts.items():
    ...     print(f"{zone_name}: {len(gdf)} feature(s)")
    resting_area: 1 feature(s)
    flat_trail: 2 feature(s)
    stairs: 2 feature(s)
    ramp: 2 feature(s)
    """
    data_directory = Path(data_directory)

    layout_files = {
        'resting_area': 'resting area.geojson',
        'flat_trail': 'flat trail.geojson',
        'stairs': 'stairs.geojson',
        'ramp': 'ramp.geojson'
    }

    result = {}

    for zone_name, filename in layout_files.items():
        file_path = data_directory / filename

        if file_path.exists():
            result[zone_name] = load_spatial_layout(file_path, as_geopandas=as_geopandas)
        else:
            print(f"Warning: Layout file not found: {file_path}")

    return result


def check_point_in_zone(
    point: Tuple[float, float],
    zone_geometry: Union[gpd.GeoDataFrame, Dict],
    return_feature_name: bool = False
) -> Union[bool, Tuple[bool, Optional[str]]]:
    """
    Check if a GPS coordinate point is within a spatial zone.

    Parameters
    ----------
    point : tuple of float
        GPS coordinates as (longitude, latitude)
    zone_geometry : GeoDataFrame or dict
        Spatial zone loaded from load_spatial_layout()
    return_feature_name : bool, optional
        If True, also return the name of the feature containing the point. Default: False

    Returns
    -------
    bool or tuple
        If return_feature_name=False: Boolean indicating if point is in zone
        If return_feature_name=True: (is_in_zone, feature_name or None)

    Example
    -------
    >>> resting = load_spatial_layout('resting area.geojson')
    >>> point = (-114.13177, 51.07956)  # (longitude, latitude)
    >>> is_in_rest = check_point_in_zone(point, resting)
    >>> print(f"Point is in resting area: {is_in_rest}")
    >>>
    >>> # Get feature name
    >>> is_in, name = check_point_in_zone(point, resting, return_feature_name=True)
    >>> if is_in:
    ...     print(f"Point is in feature: {name}")

    Notes
    -----
    - Point coordinates must be in (longitude, latitude) order
    - Requires geopandas for GeoDataFrame input
    - For dict input (raw GeoJSON), uses shapely for geometry operations
    """
    if GEOPANDAS_AVAILABLE and isinstance(zone_geometry, gpd.GeoDataFrame):
        from shapely.geometry import Point
        pt = Point(point[0], point[1])

        for idx, row in zone_geometry.iterrows():
            if row['geometry'].contains(pt):
                if return_feature_name:
                    feature_name = row.get('name', None)
                    return True, feature_name
                else:
                    return True

        if return_feature_name:
            return False, None
        else:
            return False

    else:
        # Handle raw GeoJSON dict
        from shapely.geometry import shape, Point
        pt = Point(point[0], point[1])

        for feature in zone_geometry.get('features', []):
            geom = shape(feature['geometry'])
            if geom.contains(pt):
                if return_feature_name:
                    feature_name = feature.get('properties', {}).get('name', None)
                    return True, feature_name
                else:
                    return True

        if return_feature_name:
            return False, None
        else:
            return False


def identify_participant_location(
    gps_point: Tuple[float, float],
    all_layouts: Dict[str, Union[gpd.GeoDataFrame, Dict]]
) -> Optional[str]:
    """
    Identify which zone a GPS point belongs to.

    Parameters
    ----------
    gps_point : tuple of float
        GPS coordinates as (longitude, latitude)
    all_layouts : dict
        Dictionary of all layouts from load_all_spatial_layouts()

    Returns
    -------
    str or None
        Name of the zone ('resting_area', 'flat_trail', 'stairs', 'ramp')
        or None if not in any zone

    Example
    -------
    >>> layouts = load_all_spatial_layouts('path/to/layout_directory/')
    >>> point = (-114.13177, 51.07956)
    >>> location = identify_participant_location(point, layouts)
    >>> print(f"Participant is in: {location}")
    Participant is in: resting_area
    >>>
    >>> # Process GPS trajectory
    >>> contextual_data = load_contextual_sensing_data('p1-flat_ground.csv')
    >>> for idx, row in contextual_data.iterrows():
    ...     point = (row['longitude'], row['latitude'])
    ...     location = identify_participant_location(point, layouts)
    ...     if location:
    ...         print(f"{idx}: {location}")
    """
    for zone_name, zone_geom in all_layouts.items():
        if check_point_in_zone(gps_point, zone_geom):
            return zone_name

    return None
