
import numpy as np
import nibabel as nib
import os
import pandas as pd
import re
import h5py
from pathlib import Path
from datetime import datetime, timedelta


def behavior_loader(h5_file_path):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Extract data from the 'data' dataset
        dataset = file['data'][:]
        
        # Extract fields
        events = dataset['event']
        timestamps = dataset['timestamp']

    # Convert data to DataFrame
    data = []
    for i in range(len(timestamps)):
        data.append({
            'Event': events[i].decode('utf-8'),
            'Timestamp': timestamps[i]
            
        })
    print(data)
    behavior_df = pd.DataFrame(data)
    
    event_on = [event['Event'] for event in data if 'enab' in event['Event']][0]
    event_off = [event['Event'] for event in data if 'disab' in event['Event']][0]
    print(f'Using event on {event_on} and event off {event_off}')

    return calculate_stimulus_events_caltech_daq(behavior_df, event_on, event_off)



def calculate_stimulus_events_caltech_daq(behavior_df, event_on, event_off):
    """
    Analyzes a DataFrame containing behavioral event data to
    compute the durations of stimulus events.
    It extracts the times when the stimulus was turned on and off,
    calculates the duration for each stimulus event,
    and returns a DataFrame with the stimulus conditions,
    onset times, and durations.
    
    Args:
    behavior_df (DataFrame): A DataFrame with columns for timestamps,
                             event descriptions, and relative experiment times.
    
    Returns:
    DataFrame: A DataFrame containing the trial type, onset times,
               and durations of each stimulus event.
    """
    
    on_times = behavior_df[behavior_df['Event'] == event_on]['Timestamp'].reset_index(drop=True)
    off_times = behavior_df[behavior_df['Event'] == event_off]['Timestamp'].reset_index(drop=True)

    # Calculating the duration for which the stimulus was on
    if len(on_times) == len(off_times):
        stimulus_durations = off_times - on_times
    else:
        print("Mismatch in 'on' and 'off' events count.")
        stimulus_durations = off_times - on_times[0:-1]
        # stimulus_durations = stimulus_durations.append(pd.Series([stimulus_durations.mean()]))
        stimulus_durations = pd.concat([
            stimulus_durations, pd.Series([stimulus_durations.mean()])], ignore_index=True)

        # return None

    # Extract conditions and onset times
    stimulus_df = behavior_df[behavior_df['Event'] == event_on]
    conditions = stimulus_df['Event'].tolist()
    onsets = stimulus_df['Timestamp'].tolist()

    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
    )

    return events
