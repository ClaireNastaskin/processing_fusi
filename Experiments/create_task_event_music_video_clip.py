######################################
# Create task_events for clip musics #
######################################
import pandas as pd
from pathlib import Path
import numpy as np
import os

base_path = Path('/Users/clairenastaskin/data/sub-Forest001/2025-07-08/sub-Forest001_ses-20250708_task-movements_run-004')

start_clip_tsv_path = base_path / 'sub-Forest001_ses-20250708_task-movements_run-004_events.tsv'
start_clip_tsv_file = pd.read_csv(start_clip_tsv_path, sep='\t')

task_tsv_path = base_path / 'Clip5_raw_events.tsv'
task_tsv_file = pd.read_csv(task_tsv_path, sep='\t')

# Fix DataFrame initialization for mypy/pandas
task_events = pd.DataFrame({
    'trial_type': pd.Series(dtype='str'),
    'onset': pd.Series(dtype='float'),
    'duration': pd.Series(dtype='float')
})

k = 0
for trial_number in range(len(start_clip_tsv_file)):
    # Get the offset for this trial
    trial_offset = start_clip_tsv_file.loc[trial_number, 'onset']
    
    for n in range(len(task_tsv_file)):
        if task_tsv_file.loc[n, 'start']:
            # Prepare a new dict for new row, fill duration=nan for now
            new_row = {
                'trial_type': task_tsv_file.loc[n, 'task'],
                'onset': task_tsv_file.loc[n, 'timestamp_perf [s]'] + trial_offset,
                'duration': np.nan
            }
            # Find next stop index after current start index n
            ind = n + 1
            while ind < len(task_tsv_file) and not task_tsv_file.loc[ind, 'stop']:
                ind += 1
            # Calculate duration if stop found
            if ind < len(task_tsv_file):
                new_row['duration'] = (
                    task_tsv_file.loc[ind, 'timestamp_perf [s]']
                    - task_tsv_file.loc[n, 'timestamp_perf [s]']
                )
            # Append the new row
            task_events.loc[k] = new_row
            k += 1

# Remove rows where onset is negative
task_events = task_events[task_events['onset'] >= 0].copy()

# Sort by onset to ensure chronological order
task_events = task_events.sort_values('onset').reset_index(drop=True)  # type: ignore

# Save task_events and frame_times to base_path
task_events.to_csv(base_path / 'task_events.tsv', sep='\t', index=False)