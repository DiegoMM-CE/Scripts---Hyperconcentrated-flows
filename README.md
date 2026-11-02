# Image-Based Tracking of a Hyperconcentrated-Flow Front

## Overview

This repository contains five Python scripts used to obtain **front position as a function of time** for a hyperconcentrated flow moving through a channel, based on videos from multiple fixed cameras. The workflow comprises video trimming and frame extraction, visual fisheye-distortion adjustment, spatial scale calibration, HSV colour-threshold tuning, and multi-camera front tracking.

The scripts were developed for a specific experimental setup. Camera intrinsics, distortion coefficients, pixel-to-metre factors, fixed camera offsets, HSV thresholds, trim interval, and centerline selections are setup-specific. Validate these values against the actual experiment before using the code for another setup.

## Workflow and scripts

1. **`01_trim_and_extract_frames.py`** — trims videos to a configured time interval and exports JPEG frames. A channel can optionally be flipped horizontally through `MIRRORED_CHANNELS`.
2. **`02_fisheye_parameter_tuner.py`** — interactively adjusts a camera matrix and distortion coefficients while displaying a grid over the undistorted image. This is a visual tuning aid, not a substitute for a formal calibration procedure.
3. **`03_spatial_scale_calibration.py`** — calculates a pixel-to-metre factor from two image points separated by a known physical distance.
4. **`04_hsv_mask_tuner.py`** — interactively selects HSV bounds used to segment the flow from the channel background.
5. **`05_multicamera_front_tracking.py`** — processes the frame folders camera by camera, lets the user select each channel centerline, detects the front along that axis, applies a local-cluster filter, saves annotated verification frames, and writes one CSV row per attempted frame with its detection status.


## Requirements

- Python 3
- `numpy`
- `opencv-python` (the scripts use OpenCV GUI windows for interactive steps)
- `moviepy` (used by the trimming script)

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list is not yet version-pinned and has not been tested against the original video dataset. Record the exact versions used for the final research analysis. These scripts require an interactive graphical desktop for their calibration and tracking windows; they are not designed for headless execution without changes.

## Usage

Run the scripts in this general order. Edit the configuration section near the top of each file before running it.

### 1. Trim videos and extract frames

```bash
python 01_trim_and_extract_frames.py
```

Set `TARGET_FOLDER`, `START_SEC`, `END_SEC`, and `MIRRORED_CHANNELS`. The script creates a trimmed video and a corresponding `*_trimmed_frames` folder. Frame names have the form `frame_0000_time_0ms.jpg`. The timestamp is read from the trimmed video using OpenCV and is relative to the trimmed clip; verify it against the source recording and software/backend used for processing.

JPEG frames are written at quality 95. The workflow assumes the working resolution is kept consistent (the example camera matrices use a principal point near 1920 × 1080 image centre). Do not resize or crop images without revisiting the camera parameters and calibration.

### 2. Tune fisheye parameters

```bash
python 02_fisheye_parameter_tuner.py
```

Set `IMAGE_PATH` to a representative frame from the camera being calibrated. Adjust focal length, `k1`, `k2`, and `BALANCE` while inspecting physical references such as channel walls. Copy the printed `K`, `D`, and `BALANCE` into the downstream calibration/tracking configuration. Use consistent values throughout a camera's processing. Check that these visual adjustments produce adequate geometry against known references.

### 3. Calibrate spatial scale

```bash
python 03_spatial_scale_calibration.py
```

Set `IMAGE_PATH`, `REAL_DISTANCE_METERS`, `K`, `D`, and `BALANCE` for one camera. Click two marks whose separation is known. The script calculates `PIXEL_TO_METERS = REAL_DISTANCE_METERS / pixel_distance`. Repeat and verify this factor for each camera against known distances; the current method assumes one scalar factor is an adequate approximation along the relevant centerline in the undistorted image.

### 4. Tune the HSV mask

```bash
python 04_hsv_mask_tuner.py
```

Choose a representative frame and set its `K`, `D`, and `BALANCE`. Adjust the HSV sliders and inspect the mask beside the undistorted image. Copy the resulting `LOWER_MUD` and `UPPER_MUD` values into that camera's entry in `CAMERA_REGISTRY`. Check multiple frames for false positives and missed detections due to lighting, shadows, reflections, wet surfaces, or changes in flow colour.

### 5. Track front position across cameras

```bash
python 05_multicamera_front_tracking.py
```

Place the `*_trimmed_frames` folders under `TARGET_ROOT_DIRECTORY`. Review the experiment-specific values in `CAMERA_REGISTRY` for channels `ch2` to `ch6`. For every camera folder, click the start point and then the end point along the channel axis, in the intended positive direction. `camera_offset_meters` is the fixed global coordinate assigned to that camera's selected start point, and remains valid only while the camera and the physical reference geometry remain unchanged.

The front detector converts each undistorted frame to HSV, applies the camera's HSV range, samples mask-positive pixels along the selected centerline, and scans from the end of the line. A candidate is accepted when at least 12 positive samples fall within a backwards window of 16 line-sample indices. This is a heuristic noise filter and should be reviewed against manually inspected frames.

Press `e` in the preview window to stop processing the current camera early; press `q` or `Esc` during centerline selection to cancel that camera's setup.

## CSV output and detection status

The multi-camera script writes `mudflow_master_results.csv`. Its columns are:

- `Folder_Source`
- `Channel`
- `Frame_Name`
- `Timestamp_ms` — clip-relative milliseconds parsed from the frame filename
- `Global_Position_meters` — blank when no accepted front position is available
- `Detection_Status`
- `Centerline_Mask_Sample_Count` — number of mask-positive samples along the selected centerline

Statuses include:

- `detected` — an accepted front position was calculated.
- `no_mask_pixels` — no HSV-positive samples occurred along the selected centerline. This does **not** prove the flow was physically absent.
- `cluster_rejected` — positive samples existed, but none met the current cluster criterion.
- `unreadable_frame` — OpenCV could not read the image file.
- `invalid_timestamp` — the expected timestamp could not be parsed from the frame filename.

The CSV is opened in write mode, so a new run overwrites an existing output file. Back up previous results or change `MASTER_CSV_PATH` before rerunning. Only process data that may be stored in the chosen location.

## Inter-camera time synchronization

**Manual inter-camera synchronization is required before combining the camera records into a common position–time relation.** The cameras may differ by milliseconds for corresponding events. Inspect synchronized visual references, determine camera-specific time corrections, and apply/document these corrections as a separate post-processing step. The script does not automatically synchronize videos or apply time corrections, and the CSV records the frame timestamp relative to the trimmed clip.

Verify timestamps against the source videos on the actual software/backend. Document the manual synchronization procedure and any per-camera corrections used in the analysis.

## Important validation notes

- Verify the pixel-to-metre factor independently for each camera and, where practical, at more than one position along the channel.
- Confirm the values in `CAMERA_REGISTRY` correspond to the current camera mounting, resolution, distortion adjustment, HSV profile, and physical reference points.
- Verify that the images retain the resolution assumed by the calibration. The example camera matrices use a principal point near `(960, 540)` for 1920 × 1080 frames.
- Inspect the centerline, annotated frames, and CSV status distribution. The detection status is an algorithmic result and must not automatically be interpreted as physical absence or presence of flow.
- Camera offsets are fixed values for the described stationary setup. Revalidate them after any camera movement or change in the channel reference geometry.
- The scripts have not been run against the original videos here; validate output quality before using measurements in figures, tables, or scientific conclusions.

## AI-assisted development and responsibility

The scripts were developed with assistance from a generative AI tool to meet the processing requirements of the research. The resulting code was used to process data and automate steps in the investigation. 

## Academic context

- **Author:** Diego Montenegro
- **Affiliation:** Faculty of Civil Engineering, National University of Engineering (UNI), Peru
- **Thesis title:** Evaluation of Hydraulic Parameters of Hyperconcentrated Flows in the Malanche Ravine Using a Physical-Experimental Model
- **Version / release date:** Working draft - November 2026

## Citation

If this code is used or discussed in a thesis, report, or publication, cite the repository version used and reference the thesis where appropriate. Add a `CITATION.cff` file once the author metadata and release details have been finalised.
