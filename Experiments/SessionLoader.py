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


class SessionLoader:
    def __init__(self, base_path, sequence=None):
        """
        Initializes the SessionLoader object with the base path to the data and the sequence to load.

        Args:
        base_path (str): The base path to the data directory.
        sequence (str or int): The sequence to load. If None, the user will be prompted to select a sequence.

        Raises:
        ValueError: If the base path or sequence does not exist.
        """

        self.base_path = None
        self.log_file_path = None
        self.task_event_file_path = None
        self.probe_event_file_path = None
        self.sequence_data_path = None
        self.raw_data_path = None
        self.beamformed_path = None
        self.power_doppler_path = None
        self.metadata_path = None
        self.sequence = None

        # check if paths exist
        if isinstance(base_path, str):
            base_path = Path(base_path)
        if base_path.exists():
            self.base_path = base_path
        else:
            raise ValueError("The specified base path does not exist.")
        
        # Find the log file path in the logs folder
        for filename in os.listdir(self.base_path / 'logs'):
            if re.search(r'task', filename):
                self.log_file_path = self.base_path / 'logs' / filename
        if self.log_file_path is None:
            raise Warning("The log file does not exist.")
            
        # Find the event file path in the streams folder
        for filename in os.listdir(self.base_path / 'streams'):
            if re.search(r'task.*event', filename) and filename[:2] != '._':
                self.task_event_file_path = self.base_path / 'streams' / filename
            if re.search(r'probe.*event', filename) and filename[:2] != '._':
                self.probe_event_file_path = self.base_path / 'streams' / filename
        
        if self.task_event_file_path is None:
            raise Warning("The task event file does not exist.")
            
        if self.probe_event_file_path is None:
            raise Warning("The probe event file does not exist.")
            
        # Load the sequence folders
        acquisitions_path = self.base_path / 'acquisitions'
        self.sequence_folders = [d.name for d in acquisitions_path.iterdir() if d.is_dir()]
        if not self.sequence_folders:
            raise Warning("No sequence is found in the acquisitions folder.")

        # Load the sequence if provided
        if sequence is not None:
            self.load_acquisition_directories(sequence)
            
    def load_acquisition_directories(self, sequence:str = None, sequence_index:int = None):
        """
        Load the specified sequence and set the paths for raw data, beamformed, power doppler, and metadata.
        
        Args:
        sequence (str or int): The sequence to load. If an integer is provided, the sequence will be selected by index.
        
        Raises:
        ValueError: If the sequence does not exist.
        """

        # Check if sequence is an index and within the range of sequence_folders
        if sequence in self.sequence_folders:
            self.sequence = sequence
        elif sequence_index is not None and sequence_index < len(self.sequence_folders) and sequence_index >= 0:
            self.sequence = self.sequence_folders[sequence_index]
        else:
            raise ValueError(f'The specified sequence name {sequence} does not exist. Try list_acquisition_directories() to see available sequences.')
        self.sequence_data_path = self.base_path / 'acquisitions' / self.sequence
        # Check if the sequence_data_path exists

        self.raw_data_path = self.sequence_data_path / 'raw_frame_data'
        self.beamformed_path = self.sequence_data_path / 'beamformed'
        self.power_doppler_path = self.sequence_data_path / 'power_doppler'
        self.metadata_path = self.sequence_data_path / 'metadata'        

        # Check if paths exist
        if not self.raw_data_path.exists():
            raise Warning("The raw data path does not exist.")
        if not self.beamformed_path.exists():
            raise ValueError("The beamformed path does not exist.")
        if not self.power_doppler_path.exists():
            raise ValueError("The power doppler path does not exist.")
        if not self.metadata_path.exists():
            raise ValueError("The metadata path does not exist.")

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

    def extract_power_doppler_info(self):
        """
        Reads in HDF5 files in a folder, extracts filenames and timestamps and event descriptions, and organizes this information into a DataFrame.

        Args:
        power_doppler_path (str): path to folder containing power doppler HDF5 files

        Returns:
        DataFrame: A DataFrame containing the timestamps and event descriptions.
        """

        filenames = [f for f in os.listdir(self.power_doppler_path) if f.endswith('.h5') and not f.startswith('.')]
        
        # Extract datetime from filenames and store data
        data = []
        for filename in filenames:
            timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{6}', filename)
            if timestamp:
                data.append({
                    'Filename': filename,
                    'Timestamp': pd.to_datetime(timestamp.group(0).replace('-', ':'), format='%Y:%m:%dT%H:%M:%S:%f')
                })
        
        power_doppler_df=pd.DataFrame(data)
        # Sort the DataFrame by the 'Timestamp' column
        power_doppler_df = power_doppler_df.sort_values(by='Timestamp')
        
        # Reset the index of the DataFrame
        power_doppler_df.reset_index(drop=True, inplace=True)

        # Calculate the relative times since the first event
        power_doppler_df['Experiment Time'] = (power_doppler_df['Timestamp'] - power_doppler_df['Timestamp'].iloc[0]).dt.total_seconds()

        #  format excluding the date and keeping the first digit of milliseconds
        power_doppler_df['Readable Timestamp'] = power_doppler_df['Timestamp'].dt.strftime('%H:%M:%S.%f').str[:-5]

        self.power_doppler_df = power_doppler_df

        return self.power_doppler_df

    def extract_task_events(self, behavior_offset=0):
        """
        Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

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

        # Convert data to DataFrame
        data = []
        for i, t in enumerate(timestamps):
            event_decoded = events[i].decode('utf-8')
            task_type = 'SSEP'
            # audio stimulus events specific
            if event_decoded == 'start_playing' or event_decoded == 'stop_playing':
                event_decoded = 'stimulus_onset' if event_decoded == 'start_playing' else 'stimulus_offset'
                task_type = 'audio_stimulus'
            else:
                task_type = 'SSEP'
            
            if event_decoded == 'stimulus_onset' or event_decoded == 'stimulus_offset':
                try:
                    stimulus_payload = json.loads(payload[i].decode('utf-8'))
                except:
                    stimulus_payload = {'task': 'unknown'}
                if task_type == 'audio_stimulus' and "stimulus" in stimulus_payload.keys():
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
                    'Stimulus': stimulus
                })

        self.behavior_df = pd.DataFrame(data)

        self.events = self.extract_nilearn_compatible_events()
        self.events.onset = self.events.onset + behavior_offset

        return self.events

    def extract_nilearn_compatible_events(self):
        """
        Analyzes a DataFrame containing behavioral event data to compute the durations of stimulus events.
        It extracts the times when the stimulus was turned on and off, calculates the duration for each stimulus event, and returns a DataFrame with the stimulus conditions, onset times, and durations.
        
        Args:
        self.df.behavior (DataFrame): A DataFrame with columns for timestamps, event descriptions, and relative experiment times.
        
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
        conditions = stimulus_df['Event'].tolist()
        onsets = stimulus_df['Timestamp'].tolist()

        events = pd.DataFrame(
            {"trial_type": conditions, "onset": onsets, "duration": stimulus_durations}
        )

        return events

    def extract_probe_events(self):
        """
        Reads an HDF5 file, extracts timestamps and event descriptions from the dataset, and organizes this information into a DataFrame.

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
                
                data.append({
                    'Event': events[i].decode('utf-8'),
                    'raw_file_name': file_name,
                    # 'raw_output_path': payload_data["output_path"],
                    'acquisition_idx': payload_data["acquisition_idx"],
                    'time_stamp': timestamps[i] 
                })

        self.probe_events = pd.DataFrame(data)
        # Filter behavior_df for rows where the 'Event' column is 'frame_saved'

        return self.probe_events

    def process_power_doppler_files(self, num_tissue_components=None):

        filenames = [f for f in os.listdir(self.power_doppler_path) if f.endswith('.h5')]

        if not filenames:
            raise ValueError("No power doppler files found in the specified directory.")

        # Extract unique values of num_tissue_components from filenames
        unique_tissue_components = set(re.findall(r'num_tissue_components=(\d+)', ' '.join(filenames)))
        if not num_tissue_components:
            num_tissue_components = unique_tissue_components
        # Check if the desired num_tissue_components is in the unique set of tissue components available in filenames
        if str(num_tissue_components) in unique_tissue_components:
            # Filter filenames to include only those with 'num_tissue_components=N' where N is the number of tissue components
            filtered_filenames = [f for f in filenames if f'num_tissue_components={num_tissue_components}' in f]
        else:
            # Find the closest available num_tissue_components if the desired one is not available
            closest_num_tissue_components = min(unique_tissue_components, key=lambda x: abs(int(x) - num_tissue_components))
            filtered_filenames = [f for f in filenames if f'num_tissue_components={closest_num_tissue_components}' in f]
            print(f"Warning: num_tissue_components={num_tissue_components} is not available. Using closest available value: {closest_num_tissue_components}")
        
        print(f'Number of file included: {len(filtered_filenames)}/{len(filenames)}')

        # Call the function to match and add filenames
        self.match_and_add_fusi_filenames(filtered_filenames)

        # Finally, load in the actual fusi data
        fusi_data = self.load_fusi_frames()
        print(f'\nLoaded in fusi_data with shape: {fusi_data.shape}')

        return fusi_data

    def match_and_add_fusi_filenames(self, filtered_filenames):
        # Create a new column in probe_events to store the matched filenames from filtered_filenames
        
        # Check if probe_events has been extracted
        if not hasattr(self, 'probe_events'):
            self.extract_probe_events()

        self.probe_events['fusi_file_name'] = ''

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
            raise Warning("Some entries in 'fusi_file_name' are not populated.")
        else:
            print("All entries in 'fusi_file_name' are properly populated.")

        return self.probe_events
            
    def load_fusi_frames(self, frame_indices=-1):
        """
        Fetches the data from specific frames in the h5 files based on the list of frame indices.
        
        Args:
        frame_indices (list of int): The indices of the frames to fetch.
        
        Returns:
        np.array: The data from the specified frames in the h5 files, stacked along the last dimension.
        """

        # Check if probe_events has been extracted
        if not hasattr(self, 'probe_events'):
            self.extract_probe_events()

        # Check if all frame_indices are valid
        if frame_indices == -1:
            frame_indices = list(range(len(self.probe_events)))
        elif any(frame_index < 0 or frame_index >= len(self.probe_events) for frame_index in frame_indices):
            raise ValueError("One or more invalid frame indices")
        
        
        # ------------------- Load the data from the h5 files (OLD) -------------------
        # Initialize a list to hold the data arrays
        # data_list = []
        # for frame_index in frame_indices:
        #     # Get the filename for the desired frame
        #     filename = self.probe_events.iloc[frame_index]['fusi_file_name']
        #     file_path = os.path.join(self.power_doppler_path, filename)

        #     # Load the h5 file
        #     with h5py.File(file_path, 'r') as file:
        #         # Assuming the dataset name in the h5 file is 'beamformed'
        #         try:
        #             data = file['beamformed'][:]
        #         except KeyError:
        #             data = file['power_doppler'][:]
        #         data_list.append(data)
        
        # # Stack the data arrays along the last dimension
        # stacked_pd = np.stack(data_list, axis=-1)

        # N=2
        # frames_data_replicated = np.tile(stacked_pd, (1, N, 1, 1))
        # frames_data_replicated = np.flip(frames_data_replicated, axis=2) # get in the right coordinate system
        # return frames_data_replicated

        # ------------------- Load the data from the h5 files (NEW) -------------------
        self.probe_events['full_path'] = self.power_doppler_path / self.probe_events['fusi_file_name']
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
        stacked_pd = dataset_on_disk["power_doppler"].transpose("lateral", "elevation", "depth", "time").compute()
        
        self.lateral = dataset_on_disk['lateral']
        self.elevation = dataset_on_disk['elevation']
        self.depth = dataset_on_disk['depth']

        print("Loaded. releasing dataset_on_disk open-file")
        dataset_on_disk.close()
        
        return stacked_pd

    def load_nifti_from_path(self, path_name=None):
        # if NIFTI file is stored in a different location (e.g. local), provide the path
        if path_name is not None:
            output_path = path_name
        else:
            # assume folder is in the sequence_data_path
            output_path = self.fUSI_corrected_path

        if output_path.exists():
            # load NIFTI file in output_path
            for f in output_path.iterdir():
                if f.is_file() and f.suffix == '.gz' and not f.name.startswith('._'):
                    print(f'Loading NIFTI file: {f.name}')
                    imgs = self.load_nifti_file(f)
                    break
        else:
            return ValueError("The directory path does not exist.")
        if imgs:
            return ValueError("No NIFTI file found in the specified directory.")
        return imgs

    def load_nifti_file(self, filename=None):
        img = nib.load(filename)
        
        if img is None:
            raise ValueError("No NIFTI file found in the specified directory.")
        else:
            imgs = img.get_fdata()
            print(f'Loaded power doppler images stored in NIFTI. Shape: {imgs.shape}')
            return imgs

    def save_nifti_file(self, data, output_path=None, filename='original.nii.gz'):
        """
        Saves the power doppler data to a NIFTI file.
        
        Args:
        data (np.array): The power doppler data to save.
        output_path (str): The path to save the NIFTI file. If None, the file will be saved in the fUSI_corrected folder.
        
        Returns:
        str: The path to the saved NIFTI file.
        """
        if output_path is None:
            output_path = self.fUSI_corrected_path
        else:
            output_path = Path(output_path)

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

        # Create a NIFTI image
        img = nib.Nifti2Image(data, np.eye(4))
        img.to_filename(output_path / filename)
        print(f'Saved PD NIFTI in {output_path}')
        print(f'of filename: {filename}')
 
        return output_path / filename