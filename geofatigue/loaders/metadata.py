"""
Loaders for session metadata and participant demographics.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union


def load_session_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load session metadata from JSON file.

    The session metadata contains information about each experimental session including:
    - Session timing (start_time, end_time)
    - Task details (type, label, start_time, end_time)
    - Baseline health data (sleep hours, caffeine, nicotine, alcohol consumption)

    Parameters
    ----------
    file_path : str or Path
        Path to the session_metadata.json file

    Returns
    -------
    dict
        Dictionary mapping participant IDs (e.g., 'p0', 'p1', ...) to their session data.
        Each participant may have one or more sessions.

    Example
    -------
    >>> metadata = load_session_metadata('path/to/session_metadata.json')
    >>> p0_sessions = metadata['p0']
    >>> print(f"Participant p0 had {len(p0_sessions)} session(s)")
    >>> first_session = p0_sessions[0]
    >>> print(f"Session started at: {first_session['start_time']}")
    >>> print(f"Tasks performed: {[task['label'] for task in first_session['tasks']]}")
    >>> print(f"Baseline sleep hours: {first_session['baseline']['total_sleep_hours']}")

    Notes
    -----
    - Most participants have 1 session, but some (like p4) have multiple sessions
    - Task labels include: 'flat_trail', 'stairs', 'ramp'
    - All tasks are '20lbs_20min' (carrying 20 pounds for ~20 minutes)
    - Timestamps are in ISO 8601 format (UTC)
    - Baseline data may contain null values if not reported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Session metadata file not found: {file_path}")

    with open(file_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def load_demographics(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load participant demographics from CSV file.

    The demographics file contains basic characteristics for each participant including
    age, weight, height, and gender.

    Parameters
    ----------
    file_path : str or Path
        Path to the demographics.csv file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, age, weight, height, gender
        Index is set to participant ID for easy lookup

    Example
    -------
    >>> demographics = load_demographics('path/to/demographics.csv')
    >>> print(f"Total participants: {len(demographics)}")
    >>> print(f"Age range: {demographics['age'].min()}-{demographics['age'].max()} years")
    >>> print(f"Gender distribution:\\n{demographics['gender'].value_counts()}")
    >>>
    >>> # Get demographics for specific participant
    >>> p1_demo = demographics.loc['p1']
    >>> print(f"P1 - Age: {p1_demo['age']}, Weight: {p1_demo['weight']}kg, Height: {p1_demo['height']}cm")

    Notes
    -----
    - 40 participants total (p0 through p39)
    - Age range: 18-44 years
    - Weight in kilograms
    - Height in centimeters
    - Gender: 'M' (male) or 'F' (female)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Demographics file not found: {file_path}")

    demographics = pd.read_csv(file_path)

    # Set participant ID as index for easy lookup
    demographics = demographics.set_index('id')

    return demographics


def get_participant_info(
    participant_id: str,
    session_metadata: Dict[str, Any],
    demographics: pd.DataFrame
) -> Dict[str, Any]:
    """
    Get combined session and demographic information for a participant.

    Parameters
    ----------
    participant_id : str
        Participant identifier (e.g., 'p0', 'p1')
    session_metadata : dict
        Session metadata loaded from load_session_metadata()
    demographics : pd.DataFrame
        Demographics data loaded from load_demographics()

    Returns
    -------
    dict
        Combined information including sessions and demographics

    Example
    -------
    >>> metadata = load_session_metadata('session_metadata.json')
    >>> demographics = load_demographics('demographics.csv')
    >>> p1_info = get_participant_info('p1', metadata, demographics)
    >>> print(p1_info)
    {
        'participant_id': 'p1',
        'age': 42,
        'weight': 81,
        'height': 180,
        'gender': 'M',
        'num_sessions': 1,
        'sessions': [...]
    }
    """
    if participant_id not in session_metadata:
        raise ValueError(f"Participant {participant_id} not found in session metadata")

    if participant_id not in demographics.index:
        raise ValueError(f"Participant {participant_id} not found in demographics")

    demo = demographics.loc[participant_id]
    sessions = session_metadata[participant_id]

    return {
        'participant_id': participant_id,
        'age': int(demo['age']),
        'weight': int(demo['weight']),
        'height': int(demo['height']),
        'gender': demo['gender'],
        'num_sessions': len(sessions),
        'sessions': sessions
    }
