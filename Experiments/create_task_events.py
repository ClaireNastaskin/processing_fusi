######################################
# Create task_events for clip musics #
######################################
import pandas as pd
from pathlib import Path
import numpy as np
import os

base_path = Path('/Users/clairenastaskin/data/2025-05-20/sub-Forest001_ses-20250520_task-movements_run-031')
task_tsv_path = Path(base_path, '/clip1_events.tsv')
print(task_tsv_path) 
task_tsv_file = pd.read_csv(task_tsv_path, sep='\t')

task_events = pd.DataFrame(columns=['trial_type', 'onset','duration'])
k = 0
for n in range(len(task_tsv_file)):
    if task_tsv_file['start'].iloc[n]:
        task_events.loc[k] = {
            'trial_type': task_tsv_file['stimulus'].iloc[n],
            'onset': task_tsv_file['timestamp'].iloc[n]
        }
    
        # Find next stop index after current start index n
        ind = n + 1
        while ind < len(task_tsv_file) and not task_tsv_file['stop'].iloc[ind]:
            ind += 1
        
        # Calculate duration if stop found
        if ind < len(task_tsv_file):
            task_events.loc[k, 'duration'] = task_tsv_file['timestamp'].iloc[ind] - task_tsv_file['timestamp'].iloc[n]
        k += 1
        
# Replace NaN duration values with 0
task_events['duration'] = task_events['duration'].fillna(0)
task_events

# Save task_events and frame_times to base_path
task_events.to_csv(os.path.join(base_path, 'task_events.csv'), index=False)
