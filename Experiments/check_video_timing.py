import json
import cv2
import numpy as np

cap = cv2.VideoCapture("/Users/clairenastaskin/data/2025-06-03/sub-Forest001_ses-20250603_task-movements_run-009/sub-Forest001_ses-20250603_task-movements_run-009_video.mp4")

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()
video_array = np.stack(frames, axis=0)

with open("/Users/clairenastaskin/data/2025-06-03/sub-Forest001_ses-20250603_task-movements_run-009/sub-Forest001_ses-20250603_task-movements_run-009_video.json", "r") as fid:
    sidecar = json.load(fid)

print(video_array.shape, len(sidecar["FrameTiming"]))