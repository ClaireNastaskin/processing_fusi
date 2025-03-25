import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime, timezone
from typing import Union


def behavior_loader(h5_file_path, exp_start_time):
    """
    Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

    Args:
    h5_file_path (str): path to file

    Returns:
    DataFrame: A DataFrame containing the timestamps and event descriptions.
    """

    # Open the HDF5 file
    with h5py.File(h5_file_path, "r") as file:
        dataset = file["data"][:]
        events = [event.decode("utf-8") for event in dataset["event"]]
        timestamps = dataset["timestamp"]
        payloads = [payload.decode("utf-8") for payload in dataset["payload"]]
    if all(event == "event" for event in events):
        df = pd.DataFrame([json.loads(payload) for payload in payloads])
    else:
        df = pd.DataFrame({"stimulus": events})
        df["start"] = [("enab" in event or "start" in event) for event in events]
        df["stop"] = [("disab" in event or "stop" in event) for event in events]
    if "timestamp" in df.columns:
        df["timestamp"] = [
            (datetime.fromisoformat(timestamp).astimezone(timezone.utc) - exp_start_time).total_seconds()
            for timestamp in df.timestamp
        ]
    else:
        df["timestamp"] = timestamps

    df_starts = df[df["start"]]
    df_stops = df[df["stop"]]
    df = df_starts.copy().drop(columns=["start", "stop"])
    df = df.rename(columns={"stimulus": "trial_type", "timestamp": "onset"})
    if len(df) > len(df_stops):
        df = df.iloc[:-1]
    df["duration"] = np.array(df_stops["timestamp"]) - np.array(df["onset"])

    return df


def get_exp_start_time(run_folder: Path) -> Union[datetime, None]:
    commands_fpath = run_folder / "streams" / "probe-commands_stream.h5"
    if not commands_fpath.exists():
        commands_fpath = run_folder / "streams" / "probe_1-commands_stream.h5"
    if not commands_fpath.exists():
        return
    with h5py.File(commands_fpath, "r") as file:
        timestamp = file.attrs["experiment_start_utc"]
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc)


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
