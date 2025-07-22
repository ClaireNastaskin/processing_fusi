import json
import pandas as pd
from datetime import datetime

with open("/mnt/nvme-data/forest_human/sourcedata/sub-Forest001/ses-20250624/ustd/sub-Forest001_ses-20250624_task-movements_run-002_ustd.json", "r") as fid:
    configs = json.load(fid)

exp_start_time = datetime.fromisoformat(configs["exp_start_time"])

events_raw = pd.read_csv("/mnt/nvme-data/forest_human/sourcedata/sub-Forest001/ses-20250624/ustd/sub-Forest001_ses-20250624_task-movements_run-002_eventsraw.tsv", sep="\t")
t0 = datetime.fromisoformat(events_raw.iloc[0]["timestamp"])

dt = (t0 - exp_start_time).total_seconds()