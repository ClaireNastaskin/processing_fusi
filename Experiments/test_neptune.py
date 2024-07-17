
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neptune
from datetime import datetime, timedelta
from anise.gui import *
import anise.image_utils
import anise.utils
from pathlib import Path
from IPython.display import HTML, Video
from anise.SessionLoader import SessionLoader

# import nilearn as nl
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

import nibabel as nib

# Initialize Neptune

def main():

    run = neptune.init_run(
        project="forest-neurotech/auto-registration-glm",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MzgzNDNmOS1kOTRjLTQ4NDYtYWRjOC00MzgyYmYwNTJmZmYifQ==",
    )  # your credentials

    params = {"learning_rate": 0.001, "optimizer": "Adam"}
    run["parameters"] = params

    for epoch in range(10):
        run["train/loss"].append(0.9 ** epoch)

    run["eval/f1_score"] = 0.66

    run.stop()


if __name__ == "__main__":
    main()