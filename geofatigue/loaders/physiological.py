"""
Loader for physiological data from Empatica EmbracePlus AVRO files.
"""

import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
try:
    import avro.datafile
    import avro.io
except ImportError:
    avro = None


def load_avro_physiological_data(file_path: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load physiological sensor data from Empatica EmbracePlus AVRO file.

    The AVRO files contain multiple sensor streams with high-frequency physiological data.
    Each stream includes timestamps and sensor values.

    Parameters
    ----------
    file_path : str or Path
        Path to the .avro file

    Returns
    -------
    dict
        Dictionary containing sensor data with the following structure:
        {
            'accelerometer': {
                'timestamps': np.ndarray,  # Unix microseconds
                'x': np.ndarray,           # X-axis acceleration (ADC counts)
                'y': np.ndarray,           # Y-axis acceleration (ADC counts)
                'z': np.ndarray,           # Z-axis acceleration (ADC counts)
                'sampling_frequency': float,
                'conversion_params': dict  # Parameters to convert ADC to g
            },
            'gyroscope': {
                'timestamps': np.ndarray,  # Unix microseconds
                'x': np.ndarray,           # X-axis angular velocity (ADC counts)
                'y': np.ndarray,           # Y-axis angular velocity (ADC counts)
                'z': np.ndarray,           # Z-axis angular velocity (ADC counts)
                'sampling_frequency': float,
                'conversion_params': dict  # Parameters to convert ADC to rad/s
            },
            'eda': {
                'timestamps': np.ndarray,  # Unix microseconds
                'values': np.ndarray,      # Electrodermal activity (µS)
                'sampling_frequency': float
            },
            'temperature': {
                'timestamps': np.ndarray,  # Unix microseconds
                'values': np.ndarray,      # Skin temperature (°C)
                'sampling_frequency': float
            },
            'bvp': {
                'timestamps': np.ndarray,  # Unix microseconds
                'values': np.ndarray,      # Blood Volume Pulse (nW)
                'sampling_frequency': float
            },
            'systolic_peaks': {
                'timestamps': np.ndarray   # Peak timestamps (nanoseconds)
            },
            'steps': {
                'timestamps': np.ndarray,  # Unix microseconds
                'values': np.ndarray,      # Step counts
                'sampling_frequency': float
            }
        }

    Raises
    ------
    ImportError
        If avro-python3 package is not installed
    FileNotFoundError
        If the AVRO file does not exist

    Example
    -------
    >>> data = load_avro_physiological_data('path/to/file.avro')
    >>>
    >>> # Access accelerometer data
    >>> acc = data['accelerometer']
    >>> print(f"Accelerometer sampling rate: {acc['sampling_frequency']} Hz")
    >>> print(f"Number of samples: {len(acc['x'])}")
    >>>
    >>> # Convert accelerometer from ADC counts to g
    >>> params = acc['conversion_params']
    >>> acc_x_g = acc['x'] * params['conversion_factor']
    >>>
    >>> # Access EDA data
    >>> eda = data['eda']
    >>> print(f"EDA range: {eda['values'].min():.2f} - {eda['values'].max():.2f} µS")
    >>>
    >>> # Access heart rate from systolic peaks
    >>> peaks = data['systolic_peaks']['timestamps']
    >>> # Convert nanoseconds to seconds and calculate RR intervals
    >>> rr_intervals = np.diff(peaks) / 1e9  # in seconds
    >>> heart_rate = 60 / rr_intervals  # beats per minute
    >>> print(f"Mean heart rate: {heart_rate.mean():.1f} bpm")

    Notes
    -----
    - AVRO schema version: 6.5 (Empatica EmbracePlus format)
    - Accelerometer and gyroscope data are in ADC counts and require conversion
    - Use conversion_params to convert ADC counts to physical units:
        * Accelerometer: ADC counts → gravitational g
        * Gyroscope: ADC counts → rad/s
    - EDA, temperature, and BVP are already in physical units
    - Systolic peaks timestamps are in nanoseconds (different from other sensors)
    - Magnetometer and ambient light sensors are present in schema but contain no data
    - Timestamps are in UTC
    """
    if avro is None:
        raise ImportError(
            "avro-python3 is required to load AVRO files. "
            "Install it with: pip install avro-python3"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"AVRO file not found: {file_path}")

    result = {}

    with open(file_path, 'rb') as f:
        reader = avro.datafile.DataFileReader(f, avro.io.DatumReader())

        for record in reader:
            raw_data = record.get('rawData', {})

            # Extract accelerometer data
            if 'accelerometer' in raw_data and raw_data['accelerometer']:
                acc = raw_data['accelerometer']
                if acc.get('x'):  # Check if data exists
                    timestamps = _generate_timestamps(
                        acc['timestampStart'],
                        len(acc['x']),
                        acc['samplingFrequency']
                    )
                    result['accelerometer'] = {
                        'timestamps': timestamps,
                        'x': np.array(acc['x'], dtype=np.int32),
                        'y': np.array(acc['y'], dtype=np.int32),
                        'z': np.array(acc['z'], dtype=np.int32),
                        'sampling_frequency': float(acc['samplingFrequency']),
                        'conversion_params': {
                            'physical_min': acc['imuParams']['physicalMin'],
                            'physical_max': acc['imuParams']['physicalMax'],
                            'digital_min': acc['imuParams']['digitalMin'],
                            'digital_max': acc['imuParams']['digitalMax'],
                            'conversion_factor': acc['imuParams']['conversionFactor']
                        }
                    }

            # Extract gyroscope data
            if 'gyroscope' in raw_data and raw_data['gyroscope']:
                gyro = raw_data['gyroscope']
                if gyro.get('x'):  # Check if data exists
                    timestamps = _generate_timestamps(
                        gyro['timestampStart'],
                        len(gyro['x']),
                        gyro['samplingFrequency']
                    )
                    result['gyroscope'] = {
                        'timestamps': timestamps,
                        'x': np.array(gyro['x'], dtype=np.int32),
                        'y': np.array(gyro['y'], dtype=np.int32),
                        'z': np.array(gyro['z'], dtype=np.int32),
                        'sampling_frequency': float(gyro['samplingFrequency']),
                        'conversion_params': {
                            'physical_min': gyro['imuParams']['physicalMin'],
                            'physical_max': gyro['imuParams']['physicalMax'],
                            'digital_min': gyro['imuParams']['digitalMin'],
                            'digital_max': gyro['imuParams']['digitalMax'],
                            'conversion_factor': gyro['imuParams']['conversionFactor']
                        }
                    }

            # Extract EDA data
            if 'eda' in raw_data and raw_data['eda']:
                eda = raw_data['eda']
                if eda.get('values'):
                    timestamps = _generate_timestamps(
                        eda['timestampStart'],
                        len(eda['values']),
                        eda['samplingFrequency']
                    )
                    result['eda'] = {
                        'timestamps': timestamps,
                        'values': np.array(eda['values'], dtype=np.float32),
                        'sampling_frequency': float(eda['samplingFrequency'])
                    }

            # Extract temperature data
            if 'temperature' in raw_data and raw_data['temperature']:
                temp = raw_data['temperature']
                if temp.get('values'):
                    timestamps = _generate_timestamps(
                        temp['timestampStart'],
                        len(temp['values']),
                        temp['samplingFrequency']
                    )
                    result['temperature'] = {
                        'timestamps': timestamps,
                        'values': np.array(temp['values'], dtype=np.float32),
                        'sampling_frequency': float(temp['samplingFrequency'])
                    }

            # Extract BVP data
            if 'bvp' in raw_data and raw_data['bvp']:
                bvp = raw_data['bvp']
                if bvp.get('values'):
                    timestamps = _generate_timestamps(
                        bvp['timestampStart'],
                        len(bvp['values']),
                        bvp['samplingFrequency']
                    )
                    result['bvp'] = {
                        'timestamps': timestamps,
                        'values': np.array(bvp['values'], dtype=np.float32),
                        'sampling_frequency': float(bvp['samplingFrequency'])
                    }

            # Extract systolic peaks (for heart rate calculation)
            if 'systolicPeaks' in raw_data and raw_data['systolicPeaks']:
                peaks = raw_data['systolicPeaks']
                if peaks.get('peaksTimeNanos'):
                    result['systolic_peaks'] = {
                        'timestamps': np.array(peaks['peaksTimeNanos'], dtype=np.int64)
                    }

            # Extract steps data
            if 'steps' in raw_data and raw_data['steps']:
                steps = raw_data['steps']
                if steps.get('values'):
                    timestamps = _generate_timestamps(
                        steps['timestampStart'],
                        len(steps['values']),
                        steps['samplingFrequency']
                    )
                    result['steps'] = {
                        'timestamps': timestamps,
                        'values': np.array(steps['values'], dtype=np.int32),
                        'sampling_frequency': float(steps['samplingFrequency'])
                    }

        reader.close()

    return result


def _generate_timestamps(start_time_us: int, num_samples: int, sampling_freq: float) -> np.ndarray:
    """
    Generate timestamps for uniformly sampled data.

    Parameters
    ----------
    start_time_us : int
        Start timestamp in microseconds (Unix time)
    num_samples : int
        Number of samples
    sampling_freq : float
        Sampling frequency in Hz

    Returns
    -------
    np.ndarray
        Array of timestamps in microseconds
    """
    interval_us = 1e6 / sampling_freq  # Interval between samples in microseconds
    timestamps = start_time_us + np.arange(num_samples) * interval_us
    return timestamps.astype(np.int64)


def convert_adc_to_physical(
    adc_values: np.ndarray,
    conversion_params: Dict[str, Union[int, float]]
) -> np.ndarray:
    """
    Convert ADC counts to physical units using conversion parameters.

    For accelerometer: ADC counts → gravitational g
    For gyroscope: ADC counts → rad/s

    Parameters
    ----------
    adc_values : np.ndarray
        Array of ADC count values
    conversion_params : dict
        Dictionary containing 'conversion_factor' key

    Returns
    -------
    np.ndarray
        Values in physical units

    Example
    -------
    >>> data = load_avro_physiological_data('file.avro')
    >>> acc_x_adc = data['accelerometer']['x']
    >>> acc_x_g = convert_adc_to_physical(acc_x_adc, data['accelerometer']['conversion_params'])
    >>> print(f"Acceleration range: {acc_x_g.min():.2f} - {acc_x_g.max():.2f} g")
    """
    return adc_values * conversion_params['conversion_factor']


def calculate_heart_rate_from_peaks(
    peak_timestamps: np.ndarray,
    output_unit: str = 'bpm'
) -> np.ndarray:
    """
    Calculate heart rate from systolic peak timestamps.

    Parameters
    ----------
    peak_timestamps : np.ndarray
        Systolic peak timestamps in nanoseconds
    output_unit : str, optional
        Output unit: 'bpm' (beats per minute) or 'hz' (Hz)

    Returns
    -------
    np.ndarray
        Heart rate values. Length is len(peak_timestamps) - 1

    Example
    -------
    >>> data = load_avro_physiological_data('file.avro')
    >>> peaks = data['systolic_peaks']['timestamps']
    >>> hr = calculate_heart_rate_from_peaks(peaks, output_unit='bpm')
    >>> print(f"Mean HR: {hr.mean():.1f} bpm, Std: {hr.std():.1f} bpm")
    """
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(peak_timestamps) / 1e9

    # Convert to heart rate
    if output_unit == 'bpm':
        heart_rate = 60.0 / rr_intervals
    elif output_unit == 'hz':
        heart_rate = 1.0 / rr_intervals
    else:
        raise ValueError(f"Unknown output_unit: {output_unit}. Use 'bpm' or 'hz'.")

    return heart_rate


def _merge_into(target: Dict, source: Dict) -> None:
    """Merge source signal dict into target by concatenating arrays and re-sorting by timestamps.

    Scalar fields (sampling_frequency, conversion_params) are kept from the
    first file; subsequent files are assumed to use identical device settings.
    """
    for sensor, data in source.items():
        if sensor not in target:
            target[sensor] = copy.deepcopy(data)
            continue

        existing = target[sensor]
        array_fields = [k for k, v in data.items() if isinstance(v, np.ndarray)]
        for field in array_fields:
            if field in existing and isinstance(existing[field], np.ndarray):
                existing[field] = np.concatenate([existing[field], data[field]])

        if "timestamps" in existing and len(existing["timestamps"]) > 1:
            order = np.argsort(existing["timestamps"], kind="stable")
            for field in array_fields:
                if field in existing:
                    existing[field] = existing[field][order]


def load_participant_signals(
    participant_id: str,
    data_root: Union[str, Path],
) -> Dict[str, Dict]:
    """Load and merge all AVRO physiological signals for one participant.

    Scans `data_root / participant_id /` recursively for .avro files, loads
    each with load_avro_physiological_data, and concatenates signals sorted by
    timestamp.

    systolic_peaks timestamps are converted from nanoseconds (as stored in the
    raw AVRO file) to microseconds, and exposed under both the 'timestamps' and
    'values' keys so the result is compatible with extract_rr_intervals.

    Args:
        participant_id: Participant directory name, e.g. 'p0'.
        data_root: Root directory containing one sub-directory per participant.

    Returns:
        Merged signal dict with the same structure as load_avro_physiological_data,
        except systolic_peaks has both 'timestamps' and 'values' in microseconds.

    Raises:
        FileNotFoundError: If no .avro files are found for participant_id.
    """
    # Find all .avro files under data_root whose relative path contains a component
    # that equals participant_id exactly (e.g. "p0/") or starts with participant_id
    # followed by a hyphen (e.g. "p0-3YK9R1L11Y/" for hardware-ID suffixes).
    data_root = Path(data_root)
    avro_files = sorted(
        f for f in data_root.rglob("*.avro")
        if any(
            part == participant_id or part.startswith(f"{participant_id}-")
            for part in f.relative_to(data_root).parts
        )
    )
    if not avro_files:
        raise FileNotFoundError(
            f"No .avro files found for '{participant_id}' under {data_root}"
        )

    merged: Dict = {}
    for path in avro_files:
        _merge_into(merged, load_avro_physiological_data(path))

    # Convert systolic peaks nanoseconds -> microseconds; add 'values' alias
    if "systolic_peaks" in merged:
        ts_us = (merged["systolic_peaks"]["timestamps"] // 1_000).astype(np.int64)
        merged["systolic_peaks"] = {"timestamps": ts_us, "values": ts_us}

    return merged
