import os
from pathlib import Path
from anise.gui import MakeAnimation
from anise.SessionLoader import SessionLoader
import argparse

def main(root: Path, base_path: Path, save_output_locally=False):
    """
    Main function to process fUSI data and save to BIDS format, along with task info and movies
    It will process all runs within the base_path and save the output to the output_path
    
    Parameters
    ----------
    root : Path
        root path to the data (e.g. '/cassini/UCLA_collaboration/')
    base_path : Path
        base path to the data (e.g. '/2024-06-13/UCLA_006/')
        base path to the data (e.g. root / '2024-06-13/UCLA_006/')
    save_output_locally : bool, optional
         by default False: save output to root, otherwise local "Downloads" folder
    """

    runs = os.listdir(base_path)

    # define output path to save data, plots, and videos (default: /fUSI_corrected)
    # define output path to save data, plots, and videos
        output_path = Path.home() / "Downloads" / "UCLA_fUSI_BIDS"  # save to user defined location e.g. (/Downloads folder)
    else:
        # default to save in the same root folder
        output_path = root / "UCLA_fUSI_BIDS"

    # loop over runs
    for i, run in enumerate(runs):

        print('\n___________________________________________________________')
        print(f'\n\nProcessing: {run}\n')
        
        # loop over acqusitions in a run
        acqusition_id = 0 # switch between sequences if there are multiple (default: 0)
        num_sequence_folders = 1e6 # set to a large number to process all sequences, otherwise set to a specific number
        while acqusition_id < num_sequence_folders:

            #################################################################
            #          Loading paths and choose acquisition sequence 
            #################################################################

            # Find path using sessionLoader
            ses = SessionLoader(root, base_path / run, acqusition_id, output_path=output_path)
            
            num_sequence_folders = len(ses.sequence_folders)
            acqusition_id += 1
            
            #################################################################
            #                Load and save power doppler                    #
            #################################################################

            # process power doppler files based on same number of tissue components in clutter filter (PCA)
            unique_num_tissue_components = ses.find_unique_num_tissue_components_within_single_acqusition()

            # Parse separately if more than one tissue components within a single acqusition
            for n_tc in unique_num_tissue_components:
                try:
                    # filter the filenames to include only those with 'num_tissue_components=N'
                    ses.filter_power_doppler_files(num_tissue_components=n_tc)
                    
                    # load fusi data
                    fusi_data = ses.load_fusi_frames()
                except:
                    print(f'skipping this tissue component value: {n_tc}')
                    continue
            
                #################################################################
                #         Load task event and add timing offset                 #
                #################################################################

                if not hasattr(ses, 'task_events'):
                    ses.extract_task_events()
                    behavior_offset = (ses.probe_events['global_start_time'][0] - ses.task_start).total_seconds()
                    print('\n\tpwd ensemble start time:', ses.probe_events['global_start_time'][0])
                    print('-\ttask start time: \t', ses.task_start)
                    print('___________________________________________________________')
                    print(f'Adjusting behavioral offset by \t\t   {behavior_offset} seconds')

                    # adjust task events onset time
                    ses.task_events['onset'] = ses.task_events['onset'] - behavior_offset
                    ses.task_events

                #################################################################
                #                          save outputs                         #
                #################################################################

                fus_dir = ses.output_path / 'sourcedata' / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}' / 'fus'
                beh_dir = ses.output_path / 'sourcedata' / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}' / 'beh'

                # save to NIFTI and metadata output_path
                _, filename = ses.save_to_nifti(fusi_data, output_path=fus_dir)
                
                # save task events (need to run after save_to_nifti for self.output_filename)
                ses.save_task_events(output_path=beh_dir)

                #################################################################
                #           Display and save power doppler movie                #
                #################################################################

                file_path = ses.output_path / 'derivatives' / 'registration' / f'sub-{ses.subject_id}' / f'ses-{ses.session_id}'
                file_path.mkdir(parents=True, exist_ok=True)
                if fusi_data is not None:
                    ani = MakeAnimation(fusi_data[:, 0], 
                                        output_file=str( file_path / f'{filename}_before.mp4'),  # mp4
                                        fps=10, 
                                        depth=ses.sidecar['Depth'], 
                                        lateral=ses.sidecar['Lateral'], 
                                        time=ses.sidecar['VolumeTiming'],
                    )

                    ani = MakeAnimation(fusi_data[:, 0],
                                        output_file=str(file_path / f'{filename}_before.gif'), # gif
                                        fps=10, 
                                        depth=ses.sidecar['Depth'], 
                                        lateral=ses.sidecar['Lateral'], 
                                        time=ses.sidecar['VolumeTiming'],
                    )
                print(f'saved movies to {file_path}')

if __name__ == '__main__':

    # Define session base path from user
    parser = argparse.ArgumentParser()

    # Required positional arguments 
    parser.add_argument("session_dir", type=str, 
    parser.add_argument("session_dir", type=Path, 
                        default='',
                        required=True,
    
    # Optional arguments
    parser.add_argument("--root", type=str, 
    parser.add_argument("--root", type=Path, 
                        help="Root path dir up until session level, e.g. 'cassini/UCLA_collaboration/'",
                        default='cassini/UCLA_collaboration/'
                        )
    
    # Optional arguments
    parser.add_argument("--local", type=bool, 
                        help="If set True, output files will be saved to local 'Downloads/' folder. \
                            When default to false, output folder will be in same as 'root'.",
                        default=False)
    
    args = parser.parse_args()

    root = Path.home() / args.root
    base_path = root / args.session_dir
    save_output_locally = args.local # save to root
    
    print(f'Data path is set to: {base_path}')
    if save_output_locally:
        print("Output is saved in local 'Downloads/' folder rather than in the same directory as the data is stored.")

    # main(root, base_path, save_output_locally)