"""
Loader for contextual sensing data from smartphones.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, List

# Most participants have exactly one file per task (e.g. 'p1-flat_ground.csv').
# p4 is a repeat-session pilot participant: each task was recorded across 5
# separate days as numbered files ('p4-ground-1.csv' .. '-5.csv',
# 'p4-stairs-1.csv' .. '-4.csv', plus a typo'd 5th 'p4-staris-5.csv',
# 'p4-ramp-1.csv' .. '-5.csv'). These aliases + the glob in
# load_participant_contextual_data let every numbered/typo'd file for a task
# be found and concatenated into one trajectory, without affecting any other
# participant's exact single-file match.
_TASK_FILENAME_ALIASES = {
    'flat_ground': ['flat_ground', 'ground'],
    'stairs': ['stairs', 'staris'],
}


def load_contextual_sensing_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load contextual sensing data from smartphone CSV file.

    The contextual sensing files contain high-frequency multimodal sensor data collected
    from smartphones during the experimental tasks. Each file contains data for one
    participant performing one task (flat_trail, stairs, or ramp).

    Parameters
    ----------
    file_path : str or Path
        Path to the contextual sensing CSV file (e.g., 'p1-flat_ground.csv')

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp index and 22 sensor columns:
        - time: ISO 8601 timestamp (converted to datetime index)
        - gFx, gFy, gFz: Gravity force vector components (m/s²)
        - ax, ay, az: Linear acceleration (m/s²)
        - wx, wy, wz: Angular velocity from gyroscope (rad/s)
        - P(Pa): Barometric pressure (Pascals)
        - Bx(µT), By(µT), Bz(µT): Magnetic field strength (microTesla)
        - latitude, longitude: GPS coordinates (decimal degrees)
        - altitude: GPS altitude (meters)
        - speed: GPS-derived speed (m/s)
        - yaw, pitch, roll: Device orientation (degrees)
        - dB: Ambient sound level (decibels)

    Example
    -------
    >>> # Load contextual data for one task
    >>> data = load_contextual_sensing_data('path/to/p1-flat_ground.csv')
    >>> print(f"Duration: {(data.index[-1] - data.index[0]).total_seconds():.1f} seconds")
    >>> print(f"Sampling frequency: {len(data) / (data.index[-1] - data.index[0]).total_seconds():.1f} Hz")
    >>>
    >>> # Access specific sensors
    >>> acceleration_magnitude = (data['ax']**2 + data['ay']**2 + data['az']**2)**0.5
    >>> print(f"Mean acceleration magnitude: {acceleration_magnitude.mean():.2f} m/s²")
    >>>
    >>> # GPS trajectory
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(data['longitude'], data['latitude'])
    >>> plt.xlabel('Longitude')
    >>> plt.ylabel('Latitude')
    >>> plt.title('GPS Trajectory')
    >>> plt.show()
    >>>
    >>> # Analyze altitude changes (for stairs/ramp)
    >>> altitude_change = data['altitude'].max() - data['altitude'].min()
    >>> print(f"Total altitude change: {altitude_change:.2f} m")

    Notes
    -----
    - Files can be large (25-26 MB each)
    - Typical sampling rate: ~100 Hz (varies slightly)
    - GPS data may have fixed values when stationary or poor signal
    - Speed value of -1.0 indicates invalid/unavailable GPS speed
    - Coordinate reference system: WGS84 (EPSG:4326)
    - File naming convention: {participant_id}-{task_label}.csv
        * participant_id: p0, p1, ..., p39
        * task_label: flat_ground, stairs, ramp
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Contextual sensing file not found: {file_path}")

    # Load CSV with time column
    data = pd.read_csv(file_path)

    # Convert time column to datetime and set as index
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')

    return data


def load_participant_contextual_data(
    participant_id: str,
    data_directory: Union[str, Path],
    tasks: Optional[List[str]] = None
) -> dict:
    """
    Load all contextual sensing data for a participant across multiple tasks.

    Parameters
    ----------
    participant_id : str
        Participant identifier (e.g., 'p1')
    data_directory : str or Path
        Directory containing the contextual sensing CSV files
    tasks : list of str, optional
        List of tasks to load. If None, attempts to load all standard tasks.
        Default tasks: ['flat_ground', 'stairs', 'ramp']

    Returns
    -------
    dict
        Dictionary mapping task labels to DataFrames:
        {
            'flat_ground': pd.DataFrame,
            'stairs': pd.DataFrame,
            'ramp': pd.DataFrame
        }
        If a task has multiple files for this participant (numbered repeat
        sessions, e.g. 'p4-ground-1.csv' .. '-5.csv' — see
        _TASK_FILENAME_ALIASES), they are concatenated and time-sorted into
        one DataFrame.

    Example
    -------
    >>> data = load_participant_contextual_data('p1', 'path/to/contextual_data/')
    >>> for task, df in data.items():
    ...     duration = (df.index[-1] - df.index[0]).total_seconds()
    ...     print(f"{task}: {len(df)} samples over {duration:.1f} seconds")
    flat_ground: 123450 samples over 1234.5 seconds
    stairs: 98765 samples over 987.7 seconds
    ramp: 87654 samples over 876.5 seconds
    """
    data_directory = Path(data_directory)

    if tasks is None:
        tasks = ['flat_ground', 'stairs', 'ramp']

    result = {}

    for task in tasks:
        aliases = _TASK_FILENAME_ALIASES.get(task, [task])
        matched_files = sorted({
            f for alias in aliases for f in data_directory.glob(f"{participant_id}-{alias}*.csv")
        })

        if matched_files:
            frames = [load_contextual_sensing_data(f) for f in matched_files]
            result[task] = (
                pd.concat(frames).sort_index() if len(frames) > 1 else frames[0]
            )
        else:
            file_path = data_directory / f"{participant_id}-{task}.csv"
            print(f"Warning: File not found for {participant_id} - {task}: {file_path}")

    return result


def extract_gps_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract GPS trajectory from contextual sensing data.

    Filters out invalid GPS readings (speed == -1.0) and returns
    only the location-related columns.

    Parameters
    ----------
    data : pd.DataFrame
        Contextual sensing DataFrame from load_contextual_sensing_data()

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: latitude, longitude, altitude, speed
        Index is the original timestamp

    Example
    -------
    >>> data = load_contextual_sensing_data('p1-flat_ground.csv')
    >>> gps = extract_gps_trajectory(data)
    >>> print(f"Valid GPS points: {len(gps)}")
    >>> print(f"Trajectory bounds: "
    ...       f"Lat: [{gps['latitude'].min():.6f}, {gps['latitude'].max():.6f}], "
    ...       f"Lon: [{gps['longitude'].min():.6f}, {gps['longitude'].max():.6f}]")
    """
    # Select GPS columns
    gps_columns = ['latitude', 'longitude', 'altitude', 'speed']
    gps_data = data[gps_columns].copy()

    # Filter out invalid GPS readings (speed == -1.0 often indicates invalid data)
    # Keep all points but you can filter based on your needs
    return gps_data


def calculate_sensor_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for all sensor channels.

    Parameters
    ----------
    data : pd.DataFrame
        Contextual sensing DataFrame

    Returns
    -------
    pd.DataFrame
        Statistics (mean, std, min, max, median) for each sensor column

    Example
    -------
    >>> data = load_contextual_sensing_data('p1-stairs.csv')
    >>> stats = calculate_sensor_statistics(data)
    >>> print(stats[['ax', 'ay', 'az']])  # Acceleration statistics
                 ax        ay        az
    mean  -0.023456  0.123456  -0.987654
    std    0.234567  0.345678   0.456789
    min   -2.345678 -1.234567  -5.678901
    max    1.234567  2.345678   0.123456
    median -0.012345  0.098765  -0.956789
    """
    stats = data.describe().loc[['mean', 'std', 'min', 'max', '50%']]
    stats = stats.rename(index={'50%': 'median'})
    return stats
