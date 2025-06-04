##################################################################################
#   If needed: replace trial type names to remove gather similar tasks together  #
##################################################################################

import pandas as pd

task_events = pd.read_csv('/Users/clairenastaskin/data/For Tyson/sub-Forest001_ses-20250521_task-faces_run-015/sub-Forest001_ses-20250521_task-faces_run-015_events.tsv', sep='\t')


# Then replace the emotion/face mappings
task_events.loc[task_events['trial_type'].str.contains('CloseYourEyes'), 'trial_type'] = 'Control'
task_events.loc[task_events['trial_type'].str.contains('fear'), 'trial_type'] = 'Fear'
task_events.loc[task_events['trial_type'].str.contains('Fear'), 'trial_type'] = 'Fear'
# Save updated task events to TSV file
task_events.to_csv('/Users/clairenastaskin/data/For Tyson/sub-Forest001_ses-20250521_task-faces_run-015/sub-Forest001_ses-20250521_task-faces_run-015_events.tsv', sep='\t', index=False)
