import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import anise.utils
from anise.gui import MakeAnimation
from anise.SessionLoader import SessionLoader
from IPython.display import HTML, Video


def main():
    #      Define data paths and choose output path location    

    save_output_on_local = False
    root = Path('/Volumes/UCLA_collaboration/')
    base_path = root / '2024-06-13/UCLA_006/ses-2024-06-13'
    runs = os.listdir(base_path)

    # define output path to save data, plots, and videos (default: /fUSI_corrected)
    if save_output_on_local:
        output_path = Path.home() / "Downloads" / "UCLA_fUSI_BIDS"  # save to user defined location e.g. (/Downloads folder)
    else:
        # default to save in the same root folder
        output_path = root / "UCLA_fUSI_BIDS"

    for run in runs:
        print('___________________________________________________________')
        print(f'\n\nProcessing: {run}\n')
        acqusition_id = 0 # switch between sequences if there are multiple (default: 0)
        num_sequence_folders = 1e6 # set to a large number to process all sequences, otherwise set to a specific number
        while acqusition_id < num_sequence_folders:
            #################################################################
            #          Loading paths and choose acquisition sequence 

            # Find path using sessionLoader
            ses = SessionLoader(root, base_path / run, acqusition_id, output_path=output_path)
            
            num_sequence_folders = len(ses.sequence_folders)
            acqusition_id += 1
            
            #################################################################
            #                Load and save power doppler                    #
            #################################################################

            # process power doppler files based on same number of tissue components in clutter filter (PCA)
            unique_num_tissue_components = ses.find_unique_num_tissue_components_within_single_acqusition()

            # # Parse separately if more than one tissue components within a single acqusition
            # for n_tc in unique_num_tissue_components:
            #     try:
            #         ses.filter_power_doppler_files(num_tissue_components=n_tc)
                    
            #         # load fusi data
            #         fusi_data = ses.load_fusi_frames()

            #         # save to NIFTI and metadata output_path
            #         output_path, filename = ses.save_to_nifti(fusi_data, output_path=ses.output_path / 'sourcedata' 
            #                                                 / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}' / 'fus')
            #     except:
            #         continue

            # ses.extract_task_events()
            # behavior_offset = (ses.probe_events['global_start_time'][0] - ses.task_start).total_seconds()
            # print('\n\tpwd ensemble start time:', ses.probe_events['global_start_time'][0])
            # print('-\ttask start time: \t', ses.task_start)
            # print('___________________________________________________________')
            # print(f'Adjusting behavioral offset by \t\t   {behavior_offset} seconds')

            # # adjust task events onset time
            # ses.task_events['onset'] = ses.task_events['onset'] - behavior_offset
            # ses.task_events

            # # save task events
            # ses.save_task_events(output_path=ses.output_path / 'sourcedata' / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}' / 'beh')

            # #################################################################
            # #           Display and save power doppler movie                #
            # #################################################################

            # file_path = ses.output_path / 'derivatives' / 'registration' / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}'
            # file_path.mkdir(parents=True, exist_ok=True)
            # if fusi_data is not None:
            #     ani = MakeAnimation(fusi_data[:, 0], 
            #                         output_file=str( file_path / f'{filename}_before.mp4'),  # mp4
            #                         fps=10, 
            #                         depth=ses.sidecar['Depth'], 
            #                         lateral=ses.sidecar['Lateral'], 
            #                         time=ses.sidecar['VolumeTiming'],
            #     )

            #     ani = MakeAnimation(fusi_data[:, 0],
            #                         output_file=str(file_path / f'{filename}_before.gif'), # gif
            #                         fps=10, 
            #                         depth=ses.sidecar['Depth'], 
            #                         lateral=ses.sidecar['Lateral'], 
            #                         time=ses.sidecar['VolumeTiming'],
            #     )

if __name__ == '__main__':
    main()