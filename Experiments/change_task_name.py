import pandas as pd

# Read the TSV file
events_file = '/Users/clairenastaskin/data/2025-05-20/sub-Forest001_ses-20250520_task-movements_run-031/sub-Forest001_ses-20250520_task-movements_run-031_eventsraw.tsv'
df = pd.read_csv(events_file, sep='\t')

# Update trial_type column to Clip1.mp4
df['stimulus'] = 'Clip1.mp4'

# Save the updated dataframe back to TSV
df.to_csv(events_file, sep='\t', index=False)
