"""File I/O utility functions."""

import os
import re
from pathlib import Path
import scipy.io as sio


def load_data(data_dir: os.PathLike):
    
    
    # Initialize the list to hold the fUSI images
    fUSI_Images = []
    
    # Assuming N is the number of images and is known or calculated earlier in the code
    # If N is not known, you might need to dynamically find the number of files matching the pattern
    files = [f for f in os.listdir(data_dir) if re.match(r'power_dopplerW_\d+\.mat', f)]
    files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            mat_contents = sio.loadmat(file_path)
            fUSI_Images.append(mat_contents)
        else:
            print(f"File {file} not found.")
    
    # Load the experiment timing
    experiment_timing_path = os.path.join(data_dir, "experiment_timing.mat")
    if os.path.exists(experiment_timing_path):
        experiment_timing = sio.loadmat(experiment_timing_path)
    else:
        print("experiment_timing.mat not found.")
    
    return fUSI_Images, experiment_timing

def load_selected_data(data_dir: os.PathLike, selected_images: list):
    """
    Load selected fUSI images from a specified directory.

    Parameters:
    - data_dir: The directory where the .mat files are stored.
    - selected_images: A list of integers representing the indices of images to load.

    Returns:
    - A tuple containing a list of selected fUSI images and the experiment timing.
    """
    # Initialize the list to hold the selected fUSI images
    selected_fUSI_Images = []
    file_list=[]
    
    # Load the list of files and sort them
    files = [f for f in os.listdir(data_dir) if re.match(r'power_dopplerW_\d+\.mat', f)]
    files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

    for index, file in enumerate(files):
        if index in selected_images:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                mat_contents = sio.loadmat(file_path)
                file_list.append(file)
                selected_fUSI_Images.append(mat_contents)
            else:
                print(f"File {file} not found.")
    
    # Load the experiment timing
    experiment_timing_path = os.path.join(data_dir, "experiment_timing.mat")
    if os.path.exists(experiment_timing_path):
        experiment_timing = sio.loadmat(experiment_timing_path)
    else:
        print("experiment_timing.mat not found.")
    
    return selected_fUSI_Images, experiment_timing, file_list
