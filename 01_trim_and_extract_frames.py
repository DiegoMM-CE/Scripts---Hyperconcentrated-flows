"""Trim original videos and extract JPEG frames for mudflow-front tracking.

Configure TARGET_FOLDER, START_SEC, END_SEC, and MIRRORED_CHANNELS below before use.
Frame timestamps are read from the trimmed video and are relative to that clip.
"""

import os
import re
from pathlib import Path

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


# =============================================================================
# USER CONFIGURATION
# =============================================================================
TARGET_FOLDER = "."
START_SEC = 153
END_SEC = 165

# Example: ["ch3", "ch5"] if those camera channels are horizontally mirrored.
# Leave empty when no channel requires a horizontal flip.
MIRRORED_CHANNELS = []
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
JPEG_QUALITY = 95


def process_trim_and_extract(folder_path, start_time, end_time):
    """Trim eligible videos and extract all frames from the trimmed clips."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Target folder does not exist: {folder.resolve()}")
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Require 0 <= START_SEC < END_SEC.")

    video_files = sorted(
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in VIDEO_EXTENSIONS
        and "_trimmed" not in f.stem.lower()
    )

    if not video_files:
        print("No original video files found in the target folder.")
        return

    print(f"Found {len(video_files)} video(s). Starting trimming and frame extraction.")

    for input_path in video_files:
        stem = input_path.stem
        trimmed_path = input_path.with_name(f"{stem}_trimmed{input_path.suffix}")
        frame_folder = input_path.with_name(f"{stem}_trimmed_frames")

        match = re.search(r"_ch([2-6])_", input_path.name.lower())
        channel_key = f"ch{match.group(1)}" if match else None
        should_flip = channel_key in MIRRORED_CHANNELS if channel_key else False

        print("\n" + "=" * 60)
        print(f"Video: {input_path.name}")
        print(f"Requested interval: {start_time:.3f}–{end_time:.3f} s")
        if should_flip:
            print(f"Horizontal flip enabled for {channel_key}.")
        print("=" * 60)

        try:
            with VideoFileClip(str(input_path)) as video:
                if start_time >= video.duration:
                    print(
                        f"Skipping {input_path.name}: start time ({start_time}s) "
                        f"is beyond video duration ({video.duration:.3f}s)."
                    )
                    continue
                actual_end = min(float(end_time), float(video.duration))
                if actual_end < end_time:
                    print(f"End time exceeds video duration; using {actual_end:.3f} s.")
                trimmed_video = video[start_time:actual_end]
                trimmed_video.write_videofile(
                    str(trimmed_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
            print(f"Saved trimmed video: {trimmed_path.name}")
        except Exception as exc:
            print(f"Trimming failed for {input_path.name}: {exc}")
            continue

        capture = cv2.VideoCapture(str(trimmed_path))
        if not capture.isOpened():
            print(f"Could not open trimmed video: {trimmed_path}")
            capture.release()
            continue

        fps = capture.get(cv2.CAP_PROP_FPS)
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_folder.mkdir(parents=True, exist_ok=True)
        print(f"Detected FPS: {fps:.3f}; reported frame count: {expected_frames}")
        print(f"Frame output directory: {frame_folder}")

        extracted = 0
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break

                if should_flip:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                frame_name = f"frame_{extracted:04d}_time_{timestamp_ms}ms.jpg"
                frame_path = frame_folder / frame_name
                saved = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not saved:
                    print(f"Warning: failed to write frame {frame_path}")
                    continue
                extracted += 1
        except Exception as exc:
            print(f"Frame extraction failed for {trimmed_path.name}: {exc}")
        finally:
            capture.release()

        print(f"Extracted {extracted} frame(s) for {input_path.name}.")

    print("\nTrimming and frame extraction finished.")


if __name__ == "__main__":
    process_trim_and_extract(TARGET_FOLDER, START_SEC, END_SEC)
