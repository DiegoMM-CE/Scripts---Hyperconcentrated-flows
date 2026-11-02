"""Interactive visual tuner for OpenCV's fisheye undistortion model.

This is a visual adjustment tool, not a replacement for a formal camera
calibration. Inspect the channel geometry and independently validate the result.
"""

from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# USER CONFIGURATION
# =============================================================================
IMAGE_PATH = Path(
    "./XVR_ch3_main_20260505152200_20260505152600_trimmed_frames/"
    "frame_0000_time_0ms.jpg"
)
GRID_SPACING = 60
WINDOW_NAME = "Fisheye Tuner"


def nothing(_value):
    """OpenCV trackbar callback; intentionally does nothing."""


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load calibration image: {IMAGE_PATH}. Check IMAGE_PATH."
        )

    height, width = image.shape[:2]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)
    cv2.createTrackbar("Focal Length (Fx, Fy)", WINDOW_NAME, 500, 1500, nothing)
    cv2.createTrackbar("k1 (Barrel/Pincushion)", WINDOW_NAME, 500, 1000, nothing)
    cv2.createTrackbar("k2 (Edge Distortion)", WINDOW_NAME, 500, 1000, nothing)
    cv2.createTrackbar("Zoom / Crop Balance", WINDOW_NAME, 50, 100, nothing)

    print("--- FISHEYE TUNER INSTRUCTIONS ---")
    print("1. Adjust k1 and k2 while inspecting the channel walls.")
    print("2. Adjust focal length and balance to control geometry and cropping.")
    print("3. Use the red grid as a visual reference.")
    print("4. Press 'q' or ESC to print the current parameters and exit.\n")

    focal_val = 500
    k1 = 0.0
    k2 = 0.0
    balance_val = 0.5
    try:
        while True:
            focal_val = max(
                1, cv2.getTrackbarPos("Focal Length (Fx, Fy)", WINDOW_NAME)
            )
            k1_val = cv2.getTrackbarPos("k1 (Barrel/Pincushion)", WINDOW_NAME)
            k2_val = cv2.getTrackbarPos("k2 (Edge Distortion)", WINDOW_NAME)
            balance_val = (
                cv2.getTrackbarPos("Zoom / Crop Balance", WINDOW_NAME) / 100.0
            )
            k1 = (k1_val - 500) / 1000.0
            k2 = (k2_val - 500) / 1000.0

            cx, cy = width / 2.0, height / 2.0
            K = np.array(
                [[focal_val, 0.0, cx], [0.0, focal_val, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            D = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)

            try:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (width, height), np.eye(3), balance=balance_val
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
                )
                undistorted = cv2.remap(
                    image,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                preview = undistorted.copy()
                for x in range(0, width, GRID_SPACING):
                    cv2.line(preview, (x, 0), (x, height - 1), (0, 0, 255), 1)
                for y in range(0, height, GRID_SPACING):
                    cv2.line(preview, (0, y), (width - 1, y), (0, 0, 255), 1)
                cv2.imshow(WINDOW_NAME, preview)
            except cv2.error:
                # Some slider combinations may be numerically unsuitable; keep tuning.
                pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()

    cx, cy = width / 2.0, height / 2.0
    K = np.array(
        [[focal_val, 0.0, cx], [0.0, focal_val, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    D = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
    print("\n--- FINAL FISHEYE PARAMETERS ---")
    print(f"K = np.array([[{focal_val}.0, 0.0, {cx}], "
          f"[0.0, {focal_val}.0, {cy}], [0.0, 0.0, 1.0]], dtype=np.float32)")
    print(f"D = np.array([{k1}, {k2}, 0.0, 0.0], dtype=np.float32)")
    print(f"BALANCE = {balance_val:.2f}")
    print("Use the same K, D, and BALANCE values in downstream processing.")


if __name__ == "__main__":
    main()
