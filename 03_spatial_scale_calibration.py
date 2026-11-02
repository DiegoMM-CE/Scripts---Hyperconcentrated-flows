"""Calculate a pixel-to-metre scale from two known points in an image."""

from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# USER CONFIGURATION — replace with parameters for the selected camera
# =============================================================================
IMAGE_PATH = Path(
    "./XVR_ch4_main_20260514134400_20260514135100_trimmed_frames/"
    "frame_0000_time_0ms.jpg"
)
REAL_DISTANCE_METERS = 0.5
BALANCE = 0.5
K = np.array(
    [[915.0, 0.0, 960.0], [0.0, 915.0, 540.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
D = np.array([0.027, 0.062, 0.0, 0.0], dtype=np.float32)

clicked_points = []


def click_calibration_points(event, x, y, _flags, _param):
    """Record two image points selected with the left mouse button."""
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)} selected at pixel coordinates: ({x}, {y})")


def main():
    if REAL_DISTANCE_METERS <= 0:
        raise ValueError("REAL_DISTANCE_METERS must be positive.")

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    height, width = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=BALANCE
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    print("--- SPATIAL SCALE CALIBRATION ---")
    print("Click the first reference mark, then the second mark.")
    print(f"The known distance between the marks is {REAL_DISTANCE_METERS} m.")
    print("Press 'q' to cancel.")

    window_name = "Spatial Scale Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_calibration_points)
    try:
        while True:
            preview = undistorted.copy()
            for point in clicked_points:
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
            cv2.imshow(window_name, preview)

            if len(clicked_points) == 2:
                point_a, point_b = clicked_points
                pixel_distance = float(
                    np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
                )
                if pixel_distance <= 0:
                    raise ValueError("The selected calibration points must be distinct.")
                pixel_to_meters = REAL_DISTANCE_METERS / pixel_distance
                print("\n--- CALIBRATION RESULT ---")
                print(f"Pixel distance: {pixel_distance:.3f} pixels")
                print(f"PIXEL_TO_METERS = {pixel_to_meters:.9f}")
                print(f"BALANCE = {BALANCE:.2f}")
                print("Verify the resulting scale against known distances for this camera.")
                break

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Calibration cancelled.")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
