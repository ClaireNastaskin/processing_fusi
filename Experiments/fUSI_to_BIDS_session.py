import os
from pathlib import Path
from anise.gui import MakeAnimation
from anise.SessionLoader import SessionLoader
import argparse
import pandas as pd

def main(root: Path, base_path: Path, run_id, save_output_locally=False, overwrite=False, save_Bmode=False):
    """
    Main function to process fUSI data and save to BIDS format, along with task info and movies
    It will process all runs within the base_path and save the output to the out_dir
    
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

    
    if run_id:
        runs = [f'run-{run_id:02d}']
    else:
        runs = os.listdir(base_path)
    breakpoint()

    # define output path to save data, plots, and videos
    if save_output_locally:
        out_dir = Path.home() / "Downloads" / "UCLA_fUSI_BIDS"  # save to user defined location e.g. (/Downloads folder)
    else:
        # default to save in the same root folder
        out_dir = root / "UCLA_fUSI_BIDS"


    # Init scan files
    # create output directories if not exist
    if (out_dir / 'experiment_data.csv').exists() and not overwrite:
        df = pd.read_csv(out_dir / 'experiment_data.csv')
    else:
        df = pd.DataFrame(columns=[
            'power_doppler_path',
            'bmode_fname', 'power_doppler_fname',
            'bmode_img_fname', 'power_doppler_img_fname',
            'dataset', 'sub', 'run', 'sequence', 'plane', 'n_tissue_components',
            'angles', 'frequency', 'az_aperture', 'el_aperture',
            'gain', 'power_mode', 'cycles', 'metadata'])
    if (out_dir / 'excluded_data.csv').exists() and not overwrite:
        df_ex = pd.read_csv(out_dir / 'excluded_data.csv')
    else:
        df_ex = pd.DataFrame(columns=['power_doppler_path', 'reason'])
    ses_date = base_path.relative_to(root).parts[0]
    (out_dir / 'log').mkdir(parents=True, exist_ok=True)
    # loop over runs
    with open(out_dir / 'log' /f'{ses_date}_log.txt', "a") as fid:
        for i, run in enumerate(runs):

            if run.startswith('.'):
                continue
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
                ses = SessionLoader(root, base_path / run, acqusition_id, output_path=out_dir)
                
                num_sequence_folders = len(ses.sequence_folders)
                acqusition_id += 1
                seq = ses.sequence
                
                #################################################################
                #                Load and save power doppler                    #
                #################################################################

                # process power doppler files based on same number of tissue components in clutter filter (PCA)
                try:
                    unique_num_tissue_components = ses.find_unique_num_tissue_components_within_single_acqusition()
                except:
                    continue

                # Parse separately if more than one tissue components within a single acqusition
                for n_tc in unique_num_tissue_components:
                    try:
                        # filter the filenames to include only those with 'num_tissue_components=N'
                        ses.filter_power_doppler_files(num_tissue_components=n_tc)
                        
                        # load fusi data
                        fusi_data = ses.load_fusi_frames()
                    except KeyError:
                        print(f'Skipping this tissue component value: {n_tc}')
                        continue
                
                    #################################################################
                    #         Load task event and add timing offset                 #
                    #################################################################
                    if ses.task_events is None:
                        ses.extract_task_events()
                        behavior_offset = (ses.probe_events['global_start_time'][0] - ses.task_start).total_seconds()
                        print('\n\tpwd ensemble start time:', ses.probe_events['global_start_time'][0])
                        print('-\ttask start time: \t', ses.task_start)
                        print('___________________________________________________________')
                        print(f'Adjusting behavioral offset by \t\t   {behavior_offset} seconds')

                        # adjust task events onset time
                        ses.task_events['onset'] = ses.task_events['onset'] - behavior_offset
                        print(ses.task_events)
                    
                    # get sidecar json metadata
                    ses.get_sidecar_json()

                    filter = ses.sidecar['ClutterFilters'][0]['FilterType'].lower()
                    date = ses.session_id
                    sub = ses.subject_id
                    task = ses.task_name
                    run = ses.run
                    acq = ses.acqusition_id
                    plane = ses.plane
                    if ses.plane is not None:
                        bmode_base = (f'sub-{sub}_task-{task}_run-{run}_acq-{acq}_pose-{plane}')
                        pd_base = (bmode_base + f'_proc-{filter}{n_tc}ntc')
                    else:          
                        bmode_base = (f'sub-{sub}_task-{task}_run-{run}_acq-{acq}')
                        pd_base = (bmode_base + f'_proc-{filter}{n_tc}ntc')

                    # make directories
                    bmode_dir = out_dir / 'rawdata' / f'sub-{sub}' / f'ses-{date}' / 'fus'
                    bmode_img_dir = out_dir / 'derivatives' / 'bmode_images' / f'sub-{sub}' / f'ses-{date}'
                    fus_dir = out_dir / 'sourcedata' / f'sub-{sub}' / f'ses-{date}' / 'fus'
                    fus_img_dir = out_dir / 'derivatives' / 'power_doppler_images' / 'sub-{rat}' / f'ses-{date}'
                    beh_dir = out_dir / 'sourcedata' / f'sub-{sub}' / f'ses-{date}' / 'beh'
                    for this_dir in (bmode_dir, bmode_img_dir, fus_dir, fus_img_dir, beh_dir):
                        this_dir.mkdir(parents=True, exist_ok=True)

                    #################################################################
                    #           Load and save B-mode images (optional)              #
                    #################################################################

                    if save_Bmode:
                        bmode_filename = bmode_dir / (bmode_base + '_idx-0_bmode.h5')
                        bmode_img_fname = bmode_img_dir / f'{bmode_base.format(acq=acq)}_bmode.png'
                    else:
                        bmode_filename = ''
                        bmode_img_fname = ''

                    #################################################################
                    #                          save outputs                         #
                    #################################################################

                    # save to NIFTI and metadata
                    pd_filename = ses.save_to_nifti(fusi_data, fus_dir, pd_base)
                    pd_img_fname = ''

                    # save task events
                    ses.save_task_events(beh_dir, pd_base + '_events.tsv')

                    # save scan info
                    sidecar = ses.sidecar
                    df.loc[len(df.index)] = (
                        ses.power_doppler_path,
                        bmode_filename,
                        fus_dir / pd_filename,
                        bmode_img_fname, 
                        pd_img_fname,
                        date, sub, run, seq, plane, n_tc,
                        sidecar['PlaneWaveAngles'],
                        sidecar['ProbeCentralFrequency'],
                        sidecar['TxApertureMask'],
                        sidecar['TxElevationMask'],
                        sidecar['Gain'],
                        sidecar['AFE'],
                        sidecar['TxCycles'],
                        sidecar
                    )
                    df.to_csv(out_dir / 'experiment_data.csv', index=False)
                    df_ex.to_csv(out_dir / 'excluded_data.csv', index=False)
                    
                    #################################################################
                    #           Display and save power doppler movie                #
                    #################################################################

                    file_path = ses.output_path / 'derivatives' / 'registration' / f'sub-{sub}' / f'ses-{date}'
                    file_path.mkdir(parents=True, exist_ok=True)
                    if fusi_data is not None:
                        # mp4
                        ani = MakeAnimation(fusi_data[:, 0], 
                                            output_file=str( file_path / f'{pd_filename}_before.mp4'),
                                            fps=10, 
                                            depth=ses.sidecar['Depth'], 
                                            lateral=ses.sidecar['Lateral'], 
                                            time=ses.sidecar['VolumeTiming'],
                                            )
                        # gif
                        ani = MakeAnimation(fusi_data[:, 0],
                                            output_file=str(file_path / f'{pd_filename}_before.gif'),
                                            fps=10, 
                                            depth=ses.sidecar['Depth'], 
                                            lateral=ses.sidecar['Lateral'], 
                                            time=ses.sidecar['VolumeTiming'],
                                            )
                    print(f'saved movies to {file_path}')
            print(f'Finish scanning run {run}.')

if __name__ == '__main__':

    # Define session base path from user
    parser = argparse.ArgumentParser()

    # Required positional arguments 
    parser.add_argument("session_dir", type=Path,
                        help = "Session directory path, e.g. '2024-06-07/UCLA_006/'",
                        )
    
    # Optional arguments
    parser.add_argument("--root", type=Path, 
                        help="Root path dir up until session level, e.g. 'cassini/UCLA_collaboration/'",
                        default='cassini/UCLA_collaboration/'
                        )

    parser.add_argument("--run", type=int, 
                        help="Run number to process, e.g. 1",
                        default=[],
                        )
        
    # Optional arguments
    parser.add_argument("--local",
                        help="If set True, output files will be saved to local 'Downloads/' folder. \
                            When default to false, output folder will be in same as 'root'.",
                        action='store_true',
                        )
    
    parser.add_argument("--overwrite",
                        help="If set True, the scan file csvs will be overwritten.",
                        action='store_true',
                        )
    
    parser.add_argument("--bmode",
                        help="If set True, Bmode will be saved in rawdata/.",
                        action='store_true',
                        )
    
    args = parser.parse_args()
    root = Path.home() / args.root
    base_path = root / args.session_dir
    run_id = args.run
    save_output_locally = args.local # save to root
    overwrite = args.overwrite
    save_Bmode = args.bmode
    print(f'Data path is set to: {base_path}')
    if save_output_locally:
        print("Output is saved in local 'Downloads/' folder rather than in the same directory as the data is stored.")

    main(root, base_path, run_id, save_output_locally, overwrite, save_Bmode)