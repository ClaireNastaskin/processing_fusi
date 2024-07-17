import os
import re
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime, timedelta
import json
import nibabel.nifti2 as nib
import xarray as xr
from tqdm import tqdm
from silx.io.dictdump import h5todict
import anise.utils as utils

class SessionLoader:
    def __init__(self, root, base_path, run: int = None, acqusition_id: int = 0, output_path=None):
        """
        Initializes the SessionLoader object with the base path to the data and the sequence to load.

        Args:
        root (str or Path): The root directory before session name.
        base_path (str or Path): The base path to the data directory.
        run (int): The run number to load. If None, check if it's contained in base_path.
        acqusition_id (int): The sequence to load. If None, the first sequence is selected.
        output_path (str or Path): The path to save the output files. If None, the files will be saved in the base path.

        Raises:
        ValueError: If the base path or sequence does not exist.
        """

        # Initialize the paths
        self.log_file_path = None
        self.task_event_file_path = None
        self.probe_event_file_path = None
        self.sequence_data_path = None
        self.sequence_folders = None
        self.raw_data_path = None
        self.beamformed_path = None
        self.power_doppler_path = None
        self.metadata_path = None
        self.fusi_path = None
        self.output_path = output_path
        self.n_frames = 0
        self.acqusition_id = acqusition_id

        # check if paths exist
        # base_path = base_path + session_run
        if isinstance(base_path, str):
            base_path = Path(base_path)
        
        # check if run is an integer or if it is in the base_path
        run_in_base_path = utils.match_param('run', base_path)
        if run_in_base_path is None and isinstance(run, int):
            run = f'{run:02d}'
            base_path = base_path / f'run-{run}'
        else:
            run = run_in_base_path
        if run is not None:
            self.run = run
        else:
            print("No run is provided as an argument or identified from base_path. Skipping...")
            return
        
        # Human or rat subject
        self.subject_id = [part for part in base_path.relative_to(root).parts if part.startswith("UCLA")][0]
        if self.subject_id is None:
            self.subject_id = [part for part in base_path.relative_to(root).parts if part.startswith("Rat")][0]
        if self.subject_id is None:
            raise ValueError('subject_id not found')

        # get session date
        self.session_id = re.search(r'\d{4}-\d{2}-\d{2}', str(base_path)).group(0)
        if self.session_id is None:
            raise ValueError('session_id not found')

        # check if base_path exists
        if base_path.exists():
            self.base_path = base_path
        else:
            print(base_path)
            raise ValueError("The specified base path does not exist.")
        
        # Find the log file path in the logs folder
        if not (self.base_path / 'logs').exists():
            print("The log file does not exist.")
        else:
            for filename in os.listdir(self.base_path / 'logs'):
                if re.search(r'task', filename):
                    self.log_file_path = self.base_path / 'logs' / filename
            
        # Find the event file path in the streams folder
        if not (self.base_path / 'streams').exists():
            print("The streams file does not exist.")
        else:
            for filename in os.listdir(self.base_path / 'streams'):
                if re.search(r'task.*event', filename) and filename[:2] != '._':
                    self.task_event_file_path = self.base_path / 'streams' / filename
                if re.search(r'probe.*event', filename) and filename[:2] != '._':
                    self.probe_event_file_path = self.base_path / 'streams' / filename
        
        if self.task_event_file_path is None:
            print("The task event file does not exist.")
            
        if self.probe_event_file_path is None:
            print("The probe event file does not exist.")
            
        # Load the sequence folders
        acquisitions_path = self.base_path / 'acquisitions'
        if acquisitions_path.exists():
            self.sequence_folders = [d.name for d in acquisitions_path.iterdir() if d.is_dir()]
        
        # Load the sequence if provided
        if self.sequence_folders is None:
            print("No sequence is found in the given base path.")
        else:
            self.load_acquisition_directories(sequence_index=self.acqusition_id)
            
    def load_acquisition_directories(self, sequence:str = None, sequence_index:int = None):
        """
        Load the specified sequence and set the paths for raw data, beamformed, power doppler, and metadata.
        
        Args:
        sequence (str or int): The sequence name to load. If an integer is provided, the sequence will be selected by index. 
        sequence_index (int): The index of the sequence to load. If None, the sequence name will be used.
    
        Raises:
        ValueError: If the sequence does not exist.
        """

        # Check if sequence is an index and within the range of sequence_folders
        if sequence in self.sequence_folders:
            self.sequence = sequence
        elif sequence_index is not None and sequence_index < len(self.sequence_folders) and sequence_index >= 0:
            self.sequence = self.sequence_folders[sequence_index]
        else:
            print(f'Could not load any sequence. Try list_acquisition_directories() to see available sequences.')
            return
        self.sequence_data_path = self.base_path / 'acquisitions' / self.sequence

        self.raw_data_path = self.sequence_data_path / 'raw_frame_data'
        self.beamformed_path = self.sequence_data_path / 'beamformed'
        self.power_doppler_path = self.sequence_data_path / 'power_doppler'
        self.metadata_path = self.sequence_data_path / 'metadata'        
        
        self.list_acquisition_directories()

        self.print_paths()

    def list_acquisition_directories(self):
        """
        List all the directories in the acquisitions path.
        """
        if self.sequence is None:
            print("List of sequence in the acquisitions path:")
        else:
            print(f"Selected sequence in the acquisitions path:")
        # Check if sequence_folders is empty
        if len(self.sequence_folders) == 0:
            raise ValueError("No directories found in acquisitions path.")
        else:
            for i, directory in enumerate(self.sequence_folders):
                if directory == self.sequence:
                    print(f'--> {i}. {directory}')
                else:
                    print(f'    {i}. {directory}')

    def print_paths(self):
        print("\nLoaded directories:\n")
        print(f' - log_file_path:       \t{self.log_file_path}')
        print(f' - task_event_file_path:  \t{self.task_event_file_path}')
        print(f' - probe_event_file_path: \t{self.probe_event_file_path}')
        print(f' - sequence_data_path:  \t{self.sequence_data_path}')
        print(f' - raw_data_path:       \t{self.raw_data_path}')
        print(f' - beamformed_path:     \t{self.beamformed_path}')
        print(f' - power_doppler_path:  \t{self.power_doppler_path}')
        print(f' - metadata_path:       \t{self.metadata_path}')
        print(f' - output_path:         \t{self.output_path}')

    def extract_power_doppler_info(self):
        """
        Reads in HDF5 filenames in a folder, extracts filenames and timestamps and event descriptions, 
        and organizes this information into a DataFrame.

        Args:
        power_doppler_path (str): path to folder containing power doppler HDF5 files

        Returns:
        DataFrame: A DataFrame containing the timestamps and event descriptions.
        """

        filenames = [f for f in os.listdir(self.power_doppler_path) if 
                     f.endswith('.h5') and not f.startswith('.')]
        
        if not filenames:
            raise ValueError("No power doppler files found in the specified directory.")
        
        # Extract datetime from filenames and store data
        data = []
        for filename in filenames:
            timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', filename)
            if timestamp:
                data.append({
                    'Filename': filename,
                    'Timestamp': pd.to_datetime(timestamp.group(0).replace('-', ':'), 
                                                format='%Y:%m:%dT%H:%M:%S:%f')
                })
        
        power_doppler_df = pd.DataFrame(data)

        # Sort the DataFrame by the 'Timestamp' column
        power_doppler_df = power_doppler_df.sort_values(by='Timestamp')
        
        # Reset the index of the DataFrame
        power_doppler_df.reset_index(drop=True, inplace=True)

        self.power_doppler_df = power_doppler_df

        return self.power_doppler_df

    def extract_task_events(self, task_name='', task_description=''):
        """
        Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, 
        and organizes this information into a DataFrame.

        Args:
        task_event_file_path (str): path to file
        behavior_offset (int): The offset to apply to the timestamps to align with the behavioral data.

        Returns:
        DataFrame: A DataFrame containing the timestamps and event descriptions.
        """

        with h5py.File(self.task_event_file_path, 'r') as file:
            # Extract data from the 'data' dataset
            dataset = file['data'][:]
            
            # Extract fields
            events = dataset['event']
            timestamps = dataset['timestamp']
            payload = dataset["payload"]
            task_start = file.attrs['experiment_start_utc']
            self.task_start = pd.to_datetime(task_start, format='ISO8601').replace(tzinfo=None)

        # Convert data to DataFrame
        data = []
        for i, t in enumerate(timestamps):
            event_decoded = events[i].decode('utf-8')
            # audio stimulus events specific
            if event_decoded == 'start_playing' or event_decoded == 'stop_playing':
                event_decoded = 'stimulus_onset' if event_decoded == 'start_playing' else 'stimulus_offset'
                task_type = 'audio'
            else:
                task_type = 'SSEP'
            
            if event_decoded == 'stimulus_onset' or event_decoded == 'stimulus_offset':
                try:
                    stimulus_payload = json.loads(payload[i].decode('utf-8'))
                except:
                    stimulus_payload = {'task': 'unknown'}
                if task_type == 'audio' and "stimulus" in stimulus_payload.keys():
                    stimulus = stimulus_payload["stimulus"][:-4]
                elif "task" in stimulus_payload.keys():
                    stimulus = stimulus_payload["task"]
                elif "stimulus" in stimulus_payload.keys():
                    stimulus = stimulus_payload["stimulus"]
                else:
                    stimulus = 'unknown'
            
                data.append({
                    'Event': event_decoded,
                    'Timestamp': t,
                    'Stimulus': stimulus,
                    'global_time': self.task_start + timedelta(seconds=t),
                })

        # TODO: add task name and description based on data saved in self.task_event_file_path
        if not task_name and not task_description:
            if 'UCLA' in self.subject_id:
                if task_type == 'audio' and 'tone' in stimulus:
                    task_description = 'Play low tone (110 Hz) vs high tone (1760 Hz) audio stimulus'
                    task_name = 'audio' + '_tone'
                elif task_type == 'audio' and 'squeeze' in stimulus:
                    task_description = 'Play audio saying "squeeze left hand" vs reverse audio "squeeze left hand"'
                    task_name = 'audio' + '_speech'
                else:
                    task_name = 'SSEP'
                    task_description = 'Left vs right side somatosensory evoked potential'
                
            elif 'Rat' in self.subject_id:
                task_name = 'light'
                task_description = 'Blue LED flashing at 5Hz'

        self.task_name = task_name
        self.task_description = task_description

        self.behavior_df = pd.DataFrame(data)
        self.task_events = self.extract_nilearn_compatible_events()

        print('\nExtracted task events from task_event_file_path.')
        print('\nTask name:', self.task_name)
        print('\nTask description:', self.task_description)
        return self.task_events

    def extract_nilearn_compatible_events(self):
        """
        Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
        It extracts the times when the stimulus was turned on and off, calculates the duration for 
        each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
        
        Args:
        self.df.behavior (DataFrame): A DataFrame with columns for timestamps, event descriptions, 
        and relative experiment times.
        
        Returns:
        DataFrame: A DataFrame containing the trial type, onset times, and durations of each stimulus event.

        Also called calculate_stimulus_events_caltech_daq()
        """
        # Calculating the duration for which the stimulus was on
        behavior_df = self.behavior_df
        event_counts = behavior_df.groupby("Event").count()
        if event_counts.loc["stimulus_onset", "Timestamp"] != event_counts.loc["stimulus_offset", "Timestamp"]:
            print("Warning: The number of 'stimulus_onset' and 'stimulus_offset' events do not match. Dropping last event.")
            behavior_df = behavior_df.iloc[:-1]
            event_counts = behavior_df.groupby("Event").count()
        assert event_counts.loc["stimulus_onset", "Timestamp"] == event_counts.loc["stimulus_offset", "Timestamp"], \
            "The number of 'stimulus_onset' and 'stimulus_offset' events do not match. Please check the log file for errors."

        on_times = behavior_df[behavior_df['Event'] == 'stimulus_onset']['Timestamp'].reset_index(drop=True)
        off_times = behavior_df[behavior_df['Event'] == 'stimulus_offset']['Timestamp'].reset_index(drop=True)

        stimulus_durations = off_times - on_times
        # Extract conditions and onset times
        stimulus_df = behavior_df[behavior_df['Event'] == 'stimulus_onset']
        conditions = stimulus_df['Stimulus'].tolist()
        onsets = stimulus_df['Timestamp'].tolist()
        global_time_stamp = stimulus_df['global_time'].tolist()

        events = pd.DataFrame(
            {"trial_type": conditions, 
             "onset": onsets, 
             "duration": stimulus_durations, 
             "time_stamp": global_time_stamp}
        )

        return events

    def extract_probe_events(self):
        """
        Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and 
        organizes this information into a DataFrame.

        Args:
        probe_event_file_path (str): path to file

        Returns:
        probe_events: A DataFrame containing the timestamps and event descriptions.
        """

        # Open the HDF5 file
        with h5py.File(self.probe_event_file_path, 'r') as file:
            # Extract data from the 'data' dataset
            dataset = file['data'][:]
            
            # Extract fields
            events = dataset['event']
            timestamps = dataset['timestamp']
            payload = dataset['payload']

        # Convert data to DataFrame
        data = []
        for i in range(len(timestamps)):
            event = events[i].decode('utf-8')
            if event == 'frame_saved' or event == 'ensemble_saved':
                payload_data = json.loads(payload[i])

                # Extract the filename from the output_path
                file_name = os.path.basename(payload_data["output_path"])
                
                global_time = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', file_name)
                global_time = pd.to_datetime(global_time.group(0).replace('-', ':'), 
                                                    format='%Y:%m:%dT%H:%M:%S:%f')
                if i == 0:
                    start_acqusition = global_time
                
                ensemble_start_time = (global_time - start_acqusition).total_seconds()
                ensemble_end_time = timestamps[i]
                
                data.append({
                    'Event': events[i].decode('utf-8'),
                    'raw_file_name': file_name,
                    # 'raw_output_path': payload_data["output_path"],
                    'acquisition_idx': payload_data["acquisition_idx"],
                    'ensemble_start_time': ensemble_start_time,
                    'ensemble_end_time': ensemble_end_time,
                    'global_start_time': global_time,
                })

        self.probe_events = pd.DataFrame(data)
        # Filter behavior_df for rows where the 'Event' column is 'frame_saved'

        return self.probe_events

    def find_unique_num_tissue_components_within_single_acqusition(self):

        print('Scanning through power doppler folder...')
        self.power_doppler_df = self.extract_power_doppler_info()

        # Extract unique values of num_tissue_components from filenames
        self.n_tcs = set(re.findall(r'num_tissue_components=(\d+)', ' '.join(self.power_doppler_df['Filename'])))
        print(f'Unique number of tissue components found: {self.n_tcs}')
        return self.n_tcs
    
    def filter_power_doppler_files(self, num_tissue_components):
        # given the number of tissue components, filter the filenames to include only those with 
        # 'num_tissue_components=N' where N is the number of tissue components

        self.num_tissue_components = int(num_tissue_components)
        
        # Filter filenames to include only those with the specified number of tissue components
        filtered_filenames = [f for f in self.power_doppler_df['Filename'] \
                              if f'num_tissue_components={self.num_tissue_components}' in f]
        
        if filtered_filenames:
            print(f'\nNumber of file with the specified number of tissue components: {len(filtered_filenames)}/{len(self.power_doppler_df)}')        
        else:
            print(f'No files found with num_tissue_components={self.num_tissue_components}.')    

        # Call the function to match and add filenames
        self.probe_events = self.match_and_add_fusi_filenames(filtered_filenames)
        return self.probe_events
    
    def match_and_add_fusi_filenames(self, filtered_filenames):
        # Create a new column in probe_events to store the matched filenames from filtered_filenames

        print(f'\nMatching and adding filenames to probe_events...')
        
        # Check if probe_events has been extracted
        self.extract_probe_events()

        self.probe_events['fusi_file_name'] = None

        # Iterate through each row in probe_events
        for index, row in self.probe_events.iterrows():
            # Extract the filename from the current row in probe_events
            probe_filename = row['raw_file_name']
            # Check if this filename exists in the filtered_filenames list
            matched_filenames = [filename for filename in filtered_filenames if probe_filename[0:-3] in filename]

            # If there is a match, add the filename to the new column
            if matched_filenames:
                self.probe_events.at[index, 'fusi_file_name'] = matched_filenames[0]

        # Check if every entry in 'fusi_file_name' column is populated
        if self.probe_events['fusi_file_name'].isnull().any():
            print("Some entries in 'fusi_file_name' are not populated. See more in 'self.probe_events'.")
        else:
            print("All entries in 'fusi_file_name' are properly populated.")

        self.probe_events['full_path'] = self.power_doppler_path / self.probe_events['fusi_file_name']

        self.n_frames = len(self.probe_events)
        return self.probe_events
            
    def load_fusi_frames(self, frame_indices=-1):
        """
        Fetches the data from specific frames in the h5 files based on the list of frame indices.
        
        Args:
        frame_indices (list of int): The indices of the frames to fetch.
        
        Returns:
        imgs (np.array): The data from the specified frames in the h5 files, stacked along the last dimension.
        """

        # Check if probe_events has been extracted
        if not hasattr(self, 'probe_events'):
            self.extract_probe_events()

        # Check if all frame_indices are valid
        if frame_indices == -1:
            frame_indices = list(range(self.n_frames))
        elif any(frame_index < 0 or frame_index >= len(self.probe_events) for frame_index in frame_indices):
            raise ValueError("One or more invalid frame indices")

        #  Load the data from the h5 files
        if self.probe_events['fusi_file_name'].isnull().all():
            print("'fusi_file name' is not properly populated. Please check the filenames.")
            return None
        
        dataset_on_disk = xr.open_mfdataset(
            paths=self.probe_events.loc[frame_indices]['full_path'],
            engine="h5netcdf",
            combine="nested",
            concat_dim="time",
            coords="all",
        )

        # in NIFTI files, dimensions 1, 2, 3 are for space, dimension 4 is for time
        # order the dimensions to be (lateral, elevation, depth, time) to match NIFTI files xyz coordinates
        print("Loading dataset from disk to memory")
        fusi_data = dataset_on_disk["power_doppler"].transpose("lateral", "elevation", "depth", "time").compute()

        print("Loaded. releasing dataset_on_disk open-file")
        dataset_on_disk.close()
        
        if fusi_data.shape[1] == 1:
            N=2 # duplicate the data along the elevation dimension
            frames_data_replicated = np.tile(fusi_data, (1, N, 1, 1))
            print(f'\nLoaded in fusi_data with shape: {frames_data_replicated.shape}')
            return frames_data_replicated
        else:
            print(f'\nLoaded in fusi_data with shape: {fusi_data.shape}')
            return fusi_data

    def load_nifti(self, path_name=None, filename=None):
        """
        load the power doppler data from NIFTI file.
        
        Args:
        path_name (str): The path to load the NIFTI file.
        filename (str): The filename to load the NIFTI file.
        
        Returns:
        imgs (np.array): The data from the specified frames in the h5 files, stacked along the last dimension.

        """
        # if NIFTI file is stored in a different location (e.g. local), provide the path
        if path_name is not None:
            output_path = path_name
        else:
            # assume folder is in the sequence_data_path
            output_path = self.output_path
        
        if filename is not None:
            full_filename = output_path / filename
        else:
            full_filename = None
        
        # load NIFTI file in output_path
        imgs = None
        if full_filename is None:
            print(f"\nNo filename is given. Looking for NIFTI file in {output_path}.")
            try:
                for f in output_path.iterdir():
                    if f.is_file() and f.suffix == '.gz' and not f.name.startswith('._'):
                        print(f'\nLoading NIFTI file: {f.name}')
                        img = nib.load(f)
                        imgs = img.get_fdata()
                        break
            except FileNotFoundError:
                print(f'No NIFTI file found in {output_path}.')
        elif full_filename.exists():
            img = nib.load(full_filename)
            imgs = img.get_fdata()            
        
        assert imgs is not None, "No NIFTI file was found."
        print(f'\nLoaded power doppler images stored in NIFTI. Shape: {imgs.shape}')
        
        if self.n_frames == 0:
            self.n_frames = imgs.shape[-1]
        
        return imgs

    ### Save functions
    def save_to_nifti(self, data, output_path=None, filename_tag=''):
        """
        Saves the power doppler data to a NIFTI file.
        
        Args:
        data (np.array): The power doppler data to save.
        output_path (str): The path to save the NIFTI file. If None, the file will be saved in the output folder.
        
        Returns:
        output_path: The path to the saved NIFTI file.
        filename: The name of the saved NIFTI file ()

        """
        if output_path is None:
            output_path = self.output_path
        else:
            output_path = Path(output_path)
                
        if output_path is not None and not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print('\nWARNING: You DO NOT have write access to the base path. Please provide a different output path.')
                output_path = None
        
        if data is None:
            print('\nNo data to save. \nExiting...')
            return None
        
        # affine to rescale and flip the images
        affine=np.array([[0.2, 0.,  0.,  0.],
                        [0.,  0.2, 0.,  0.],
                        [0.,  0., -0.2, 0.],
                        [0.,  0.,  0.,  0.]])

        # Get the header
        hdr = nib.Nifti2Header()
        hdr['descrip'] = '"lateral", "elevation", "depth", "time"'

        # define the nifti image
        nifti_img = nib.Nifti2Image(data, 
                                    affine=affine,
                                    header=hdr)
        
        # get sidecar json metadata
        self.get_sidecar_json()
        
        # save the nifti file
        if filename_tag:
            filename_tag = '_' + filename_tag
        
        filter = self.sidecar['ClutterFilters'][0]['FilterType'].lower()

        if hasattr(self, 'plane'):
            self.output_filename = (f'sub-{self.subject_id}_'
                        f'task-{self.task_name}_'
                        f'run-{self.run}_'
                        f'acq-{self.acqusition_id}_'
                        f'pose-{self.plane}_'
                        f'proc-{filter}_{self.num_tissue_components}'
                        )
        else:          
            self.output_filename = (f'sub-{self.subject_id}_'
                        f'task-{self.task_name}_'
                        f'run-{self.run}_'
                        f'acq-{self.acqusition_id}_'
                        f'proc-{filter}_{self.num_tissue_components}'
                        )
            
        filename = self.output_filename + '_pwdt' + filename_tag
        nifti_img.to_filename(output_path / (filename + '.nii.gz'))
        print(f'Output location: {output_path}')
        print(f'Saved PD NIFTI file as: {filename}.nii.gz')
        
        # save metadata as a json
        self.save_sidecar_json(output_path, filename)
        return output_path, filename

    def get_sidecar_json(self):
        # TODO: add bmode dataframe by reading in all the bmode files
        if hasattr(self, 'bmode_df'):
            self.bmode_df = bmode_h5
        
        # now its any beamformed file not particularly the first
        bmode_h5 = next(self.beamformed_path.glob("[!.]*.h5"))

        # get first power doppler file
        pwd_h5 = self.probe_events['full_path'][0]
        
        time_stamps = self.probe_events['ensemble_start_time'] # in seconds
        
        n_tc = int(utils.get_param('num_tissue_components', str(self.probe_events['fusi_file_name'][0])))
    

        if 'UCLA' in self.subject_id:
            institution_name = 'UCLA'
            institution_address = '760 Westwood Plaza, Los Angeles, CA 90095'
            institution_dept = 'Department of Neurology'
        elif 'Rat' in self.subject_id:
            institution_name = 'Caltech'
            institution_address = '1200 E California Blvd, Pasadena, CA 91125'
            institution_dept = 'Brain Imaging Center'
        else:
            institution_name = ''
            institution_address = ''
            institution_dept = ''

        self.sidecar = utils.get_sidecar(bmode_h5, 
                                         pwd_h5, 
                                         time_stamps, 
                                         n_tc,
                                         task_name=self.task_name,
                                         task_description=self.task_description,
                                         institution_name=institution_name,
                                         institution_address=institution_address,
                                         institutional_department_name=institution_dept)

    def save_sidecar_json(self, output_path, filename):
        # Save metadata as a json if it exists
        if self.sidecar is not None:
            with open(output_path / f'{filename}.json', 'w') as f:
                f.write(json.dumps(self.sidecar, indent=4))
        print(f'Saved sidecar json as: {filename}.json')
    
    def save_task_events(self, output_path=None):
        # save probe_events to csv
        if output_path is not None and not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print('\nWarning: You DO NOT have write access to the base path. Please provide a different output path.')
                output_path = None
                return
        filename = self.output_filename + '_events.tsv'
        self.task_events.to_csv(output_path / filename, sep='\t', index=False)
        print(f'Output location: {output_path}')
        print(f'Saved task events as:   {filename}')
