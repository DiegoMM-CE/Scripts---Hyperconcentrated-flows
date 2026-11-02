"""Track hyperconcentrated-flow front position across multiple camera folders.

One CSV row is written for each frame that the script attempts to process.
Camera timestamps are clip-relative; manually check and correct inter-camera
synchronization before combining camera records into a common time series.
"""

import csv
import os
import re
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# CAMERA PROFILES — values reflect the experiment-specific configuration
# Verify the intrinsics, scale, offsets, and HSV bounds for every camera.
# =============================================================================
CAMERA_REGISTRY = {
    "ch2": {
        "K": np.array([[930.0, 0.0, 960.0], [0.0, 930.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "D": np.array([0.065, 0.05, 0.0, 0.0], dtype=np.float32),
        "pixel_to_meters": 0.0003968,
        "camera_offset_meters": 0.0,
        "lower_mud": np.array([5, 60, 70], dtype=np.uint8),
        "upper_mud": np.array([26, 135, 135], dtype=np.uint8),
    },
    "ch3": {
        "K": np.array([[699.0, 0.0, 960.0], [0.0, 699.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "D": np.array([0.299, 0.0, 0.0, 0.0], dtype=np.float32),
        "pixel_to_meters": 0.0003944,
        "camera_offset_meters": 0.45,
        "lower_mud": np.array([0, 30, 43], dtype=np.uint8),
        "upper_mud": np.array([26, 129, 124], dtype=np.uint8),
    },
    "ch4": {
        "K": np.array([[915.0, 0.0, 960.0], [0.0, 915.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "D": np.array([0.027, 0.062, 0.0, 0.0], dtype=np.float32),
        "pixel_to_meters": 0.0004115,
        "camera_offset_meters": 1.15,
        "lower_mud": np.array([7, 47, 25], dtype=np.uint8),
        "upper_mud": np.array([27, 188, 124], dtype=np.uint8),
    },
    "ch5": {
        "K": np.array([[1181.0, 0.0, 960.0], [0.0, 1181.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "D": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "pixel_to_meters": 0.0004077,
        "camera_offset_meters": 1.65,
        "lower_mud": np.array([6, 55, 63], dtype=np.uint8),
        "upper_mud": np.array([26, 129, 138], dtype=np.uint8),
    },
    "ch6": {
        "K": np.array([[1105.0, 0.0, 960.0], [0.0, 1105.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "D": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "pixel_to_meters": 0.0004405,
        "camera_offset_meters": 2.20,
        "lower_mud": np.array([7, 60, 63], dtype=np.uint8),
        "upper_mud": np.array([26, 130, 145], dtype=np.uint8),
    },
}

BALANCE = 0.5
TARGET_ROOT_DIRECTORY = "."
MASTER_CSV_PATH = "mudflow_master_results.csv"
VERIFICATION_SUFFIX = "_VERIFICATION"

clicked_points = []
setup_done = False
setup_cancelled = False

CSV_FIELDS = [
    "Folder_Source",
    "Channel",
    "Frame_Name",
    "Timestamp_ms",
    "Global_Position_meters",
    "Detection_Status",
    "Centerline_Mask_Sample_Count",
]


def click_event(event, x, y, _flags, _param):
    """Register the start and end of the camera-specific channel centerline."""
    global clicked_points, setup_done
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)} registered at pixel coordinates: ({x}, {y})")
        if len(clicked_points) == 2:
            setup_done = True


def build_centerline(sample_image, channel_key):
    """Interactively select the axis used to sample the segmentation mask."""
    global clicked_points, setup_done, setup_cancelled
    clicked_points = []
    setup_done = False
    setup_cancelled = False

    height, width = sample_image.shape[:2]
    cfg = CAMERA_REGISTRY[channel_key]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cfg["K"], cfg["D"], (width, height), np.eye(3), balance=BALANCE
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        cfg["K"], cfg["D"], np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )
    undistorted_sample = cv2.remap(
        sample_image, map1, map2, interpolation=cv2.INTER_LINEAR
    )

    window_name = f"Setup Centerline - {channel_key.upper()}"
    print(f"\nSelect START then END of the channel centerline for {channel_key.upper()}.")
    print("Click two points in the positive flow direction. Press q or ESC to cancel.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)
    try:
        while not setup_done:
            preview = undistorted_sample.copy()
            for point in clicked_points:
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
            if len(clicked_points) == 1:
                cv2.putText(preview, "Select END point", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                setup_cancelled = True
                break
    finally:
        cv2.destroyWindow(window_name)

    if setup_cancelled or len(clicked_points) != 2:
        print(f"Centerline setup cancelled for {channel_key.upper()}.")
        return None

    point_start, point_end = clicked_points
    number_samples = max(abs(point_end[0] - point_start[0]), abs(point_end[1] - point_start[1]), 1)
    line_x = np.linspace(point_start[0], point_end[0], number_samples).astype(int)
    line_y = np.linspace(point_start[1], point_end[1], number_samples).astype(int)
    return point_start, point_end, line_x, line_y, map1, map2


def parse_timestamp(frame_name):
    """Extract the clip-relative millisecond timestamp from a frame filename."""
    match = re.search(r"time_(\d+)ms", frame_name)
    return int(match.group(1)) if match else None


def run_camera_folder(frame_folder, csv_writer):
    """Process one folder of frames using the corresponding camera profile."""
    frame_folder = Path(frame_folder)
    folder_name = frame_folder.name
    match = re.search(r"_ch([2-6])_", folder_name.lower())
    if not match:
        print(f"Skipping '{folder_name}': camera channel ch2–ch6 not identifiable from folder name.")
        return

    channel_key = f"ch{match.group(1)}"
    cfg = CAMERA_REGISTRY[channel_key]
    frames = sorted(
        path.name for path in frame_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not frames:
        print(f"No JPG frames found in {frame_folder}.")
        return

    sample_image = cv2.imread(str(frame_folder / frames[0]))
    if sample_image is None:
        print(f"Could not load sample frame {frame_folder / frames[0]}; skipping folder.")
        return

    centerline = build_centerline(sample_image, channel_key)
    if centerline is None:
        return
    point_start, point_end, line_x, line_y, map1, map2 = centerline

    output_dir = frame_folder.parent / f"{folder_name}{VERIFICATION_SUFFIX}"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_name = f"Live Tracking - {channel_key.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Processing {len(frames)} frame(s) from {folder_name}.")
    print(f"Verification images: {output_dir}")
    print("Press 'e' in the preview window to stop this camera early.")

    try:
        for frame_name in frames:
            timestamp_ms = parse_timestamp(frame_name)
            if timestamp_ms is None:
                csv_writer.writerow([folder_name, channel_key, frame_name, "", "", "invalid_timestamp", 0])
                print(f"Skipped {frame_name}: timestamp could not be parsed.")
                continue

            frame_path = frame_folder / frame_name
            image = cv2.imread(str(frame_path))
            if image is None:
                csv_writer.writerow([folder_name, channel_key, frame_name, timestamp_ms, "", "unreadable_frame", 0])
                print(f"Could not read {frame_path}.")
                continue

            undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
            mud_mask = cv2.inRange(hsv, cfg["lower_mud"], cfg["upper_mud"])
            line_mask_values = mud_mask[line_y, line_x]
            mud_indices = np.flatnonzero(line_mask_values == 255)
            centerline_mask_count = int(len(mud_indices))

            display_image = undistorted.copy()
            cv2.line(display_image, point_start, point_end, (255, 0, 0), 2)
            global_position = ""
            detection_status = "no_mask_pixels"
            label = "No mask-positive samples on centerline"
            label_color = (0, 0, 255)

            if centerline_mask_count > 0:
                true_front_index = None
                # Scan from the end of the selected axis and reject isolated mask noise.
                for index in np.sort(mud_indices)[::-1]:
                    cluster_count = int(
                        np.sum((mud_indices >= index - 15) & (mud_indices <= index))
                    )
                    if cluster_count >= 12:
                        true_front_index = int(index)
                        break

                if true_front_index is None:
                    detection_status = "cluster_rejected"
                    label = "Candidate rejected by cluster filter"
                else:
                    front_x = int(line_x[true_front_index])
                    front_y = int(line_y[true_front_index])
                    pixel_distance = float(
                        np.hypot(front_x - point_start[0], front_y - point_start[1])
                    )
                    position = cfg["camera_offset_meters"] + (
                        pixel_distance * cfg["pixel_to_meters"]
                    )
                    global_position = round(float(position), 4)
                    detection_status = "detected"
                    label = f"Front position: {position:.3f} m"
                    label_color = (0, 255, 0)
                    cv2.circle(display_image, (front_x, front_y), 6, (0, 255, 0), -1)

            csv_writer.writerow([
                folder_name,
                channel_key,
                frame_name,
                timestamp_ms,
                global_position,
                detection_status,
                centerline_mask_count,
            ])
            print(
                f"[{channel_key.upper()}] t={timestamp_ms} ms | "
                f"status={detection_status} | position={global_position}"
            )
            cv2.putText(
                display_image,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                label_color,
                2,
            )
            cv2.imshow(window_name, display_image)
            cv2.imwrite(str(output_dir / f"v_{frame_name}"), display_image)

            if cv2.waitKey(1) & 0xFF == ord("e"):
                print(f"Early stop requested for {channel_key.upper()}.")
                break
    finally:
        cv2.destroyWindow(window_name)

    print(f"Finished {channel_key.upper()}; output rows include detection status for processed frames.")


def main():
    root_dir = Path(TARGET_ROOT_DIRECTORY)
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Target root directory does not exist: {root_dir.resolve()}")

    target_folders = sorted(
        path for path in root_dir.iterdir()
        if path.is_dir() and "_trimmed_frames" in path.name
    )
    print(f"Found {len(target_folders)} candidate frame folder(s) in {root_dir.resolve()}.")

    output_csv = Path(MASTER_CSV_PATH)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_FIELDS)
        for folder in target_folders:
            run_camera_folder(folder, writer)
            csv_file.flush()

    print(f"\nMulti-camera processing finished. CSV written to: {output_csv.resolve()}")
    print("Reminder: manually verify and correct inter-camera synchronization before combining time series.")


if __name__ == "__main__":
    main()
