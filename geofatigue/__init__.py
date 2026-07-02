"""
GeoFatigue Dataset Utilities

A Python package for loading and working with the GeoFatigue dataset -
a comprehensive multimodal dataset for fatigue monitoring in construction workers.

The dataset includes:
- Physiological data from Empatica EmbracePlus wearables (accelerometer, gyroscope, EDA, BVP, temperature, steps)
- Contextual sensing data from smartphones (IMU, GPS, barometric pressure, magnetometer, sound)
- Spatial data defining the experiment site layout (GeoJSON polygons and linestrings)
- Session metadata and participant demographics

Example usage:
    >>> from geofatigue.loaders import load_session_metadata, load_demographics
    >>> sessions = load_session_metadata('path/to/session_metadata.json')
    >>> demographics = load_demographics('path/to/demographics.csv')
"""

__version__ = '1.0.0'
__author__ = 'Mahnoush Jahromi'

from geofatigue.loaders import (
    load_session_metadata,
    load_demographics,
    load_avro_physiological_data,
    load_contextual_sensing_data,
    load_spatial_layout,
)

__all__ = [
    'load_session_metadata',
    'load_demographics',
    'load_avro_physiological_data',
    'load_contextual_sensing_data',
    'load_spatial_layout',
]
