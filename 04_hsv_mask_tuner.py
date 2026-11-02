"""Interactively tune HSV thresholds to segment the hyperconcentrated flow."""

from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# USER CONFIGURATION — replace with a representative frame for each camera
# =============================================================================
IMAGE_PATH = Path(
    "./XVR_ch6_main_20260519155500_20260519160000_trimmed_frames/"
    "frame_0071_time_2366ms.jpg"
)
BALANCE = 0.5
K = np.array(
    [[1105.0, 0.0, 960.0], [0.0, 1105.0, 540.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
WINDOW_NAME = "Mudflow Color Tuner"


def nothing(_value):
    """OpenCV trackbar callback; intentionally does nothing."""


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}. Check IMAGE_PATH before running."
        )

    height, width = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=BALANCE
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.createTrackbar("Lower H", WINDOW_NAME, 0, 180, nothing)
    cv2.createTrackbar("Lower S", WINDOW_NAME, 0, 255, nothing)
    cv2.createTrackbar("Lower V", WINDOW_NAME, 30, 255, nothing)
    cv2.createTrackbar("Upper H", WINDOW_NAME, 180, 180, nothing)
    cv2.createTrackbar("Upper S", WINDOW_NAME, 80, 255, nothing)
    cv2.createTrackbar("Upper V", WINDOW_NAME, 120, 255, nothing)

    print("--- HSV THRESHOLD TUNER ---")
    print("Adjust sliders and inspect the mask displayed next to the frame.")
    print("Press 'q' or ESC to print the current thresholds and exit.")

    values = (0, 0, 30, 180, 80, 120)
    try:
        while True:
            values = (
                cv2.getTrackbarPos("Lower H", WINDOW_NAME),
                cv2.getTrackbarPos("Lower S", WINDOW_NAME),
                cv2.getTrackbarPos("Lower V", WINDOW_NAME),
                cv2.getTrackbarPos("Upper H", WINDOW_NAME),
                cv2.getTrackbarPos("Upper S", WINDOW_NAME),
                cv2.getTrackbarPos("Upper V", WINDOW_NAME),
            )
            lh, ls, lv, uh, us, uv = values
            lower_bound = np.array([lh, ls, lv], dtype=np.uint8)
            upper_bound = np.array([uh, us, uv], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            preview = np.hstack((undistorted, mask_bgr))
            cv2.imshow(WINDOW_NAME, preview)

            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()

    lh, ls, lv, uh, us, uv = values
    print("\n--- SELECTED HSV THRESHOLDS ---")
    print(f"LOWER_MUD = np.array([{lh}, {ls}, {lv}], dtype=np.uint8)")
    print(f"UPPER_MUD = np.array([{uh}, {us}, {uv}], dtype=np.uint8)")
    print(f"BALANCE = {BALANCE:.2f}")
    if lh > uh or ls > us or lv > uv:
        print("WARNING: At least one lower HSV bound exceeds its upper bound; the mask may be empty.")


if __name__ == "__main__":
    main()
