######################################
# Create task_events for clip musics #
######################################
import pandas as pd
from pathlib import Path
import numpy as np
import os

base_path = Path('/Users/clairenastaskin/data/2025-06-03/sub-Forest001_ses-20250603_task-movements_run-009')  

start_clip_tsv_path = base_path / 'sub-Forest001_ses-20250603_task-movements_run-009_events.tsv'
start_clip_tsv_file = pd.read_csv(start_clip_tsv_path, sep='\t')

task_tsv_path = base_path / 'raw_events_for_clip1.tsv'
task_tsv_file = pd.read_csv(task_tsv_path, sep='\t')

task_events = pd.DataFrame(columns=['trial_type', 'onset','duration'])
k = 0
for n in range(len(task_tsv_file)):
    if task_tsv_file['start'].iloc[n]:
        task_events.loc[k] = {
            'trial_type': task_tsv_file['task'].iloc[n],
            'onset': task_tsv_file['timestamp_perf [s]'].iloc[n]
        }
    
        # Find next stop index after current start index n
        ind = n + 1
        while ind < len(task_tsv_file) and not task_tsv_file['stop'].iloc[ind]:
            ind += 1
        
        # Calculate duration if stop found
        if ind < len(task_tsv_file):
            task_events.loc[k, 'duration'] = task_tsv_file['timestamp_perf [s]'].iloc[ind] - task_tsv_file['timestamp_perf [s]'].iloc[n]
        k += 1
        
# task_events['onset'] = task_events['onset'] + start_clip_tsv_file['onset'].iloc[0]
task_events['onset'] = task_events['onset'] - (95.51 + 2.8)
# Remove rows where onset is negative
task_events = task_events[task_events['onset'] >= 0]


task_events

# Save task_events and frame_times to base_path
task_events.to_csv(os.path.join(base_path, 'task_events.tsv'), sep='\t', index=False)