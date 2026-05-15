import pandas as pd

# Read the TSV file
events_file = '/Users/clairenastaskin/data/sub-Forest001/2026-04-09/fusi/task-ffalocalizer_run-002/sub-Forest001_ses-20260409_task-ffalocalizer_run-002_events.tsv'
df = pd.read_csv(events_file, sep='\t')

# Use regex to extract the category from patterns like "block_001_scenes.mp4"
df['trial_type'] = df['trial_type'].str.replace(r'block_\d+_scrambled_faces\.mp4', 'scrambled_faces', regex=True)
df['trial_type'] = df['trial_type'].str.replace(r'block_\d+_faces\.mp4', 'faces', regex=True)
df['trial_type'] = df['trial_type'].str.replace(r'block_\d+_scenes\.mp4', 'scenes', regex=True)

# Save the updated dataframe back to TSV
df.to_csv(events_file, sep='\t', index=False)
