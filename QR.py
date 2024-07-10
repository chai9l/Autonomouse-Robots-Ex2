import math
import cv2
import numpy as np
import pandas as pd
import time
from typing import Tuple, List

def initialize_video_capture(video_path: str) -> cv2.VideoCapture:
    """Initialize video capture from a given video path."""
    return cv2.VideoCapture(video_path)

def initialize_video_writer(output_path: str, fps: float, frame_width: int, frame_height: int) -> cv2.VideoWriter:
    """Initialize video writer to save the processed video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def initialize_aruco_detector() -> Tuple[cv2.aruco_Dictionary, cv2.aruco_DetectorParameters]:
    """Initialize ArUco dictionary and detector parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_parameters = cv2.aruco.DetectorParameters()
    return aruco_dict, aruco_parameters

def calculate_yaw_pitch_roll(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a rotation matrix to Euler angles (yaw, pitch, roll).

    Args:
    rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
    Tuple[float, float, float]: A tuple containing the Euler angles (yaw, pitch, roll) in degrees.
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def process_frame(frame: np.ndarray, aruco_dict, aruco_parameters, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_id: int) -> List:
    """Process a single video frame to detect ArUco markers and calculate pose."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    csv_output = []

    if ids is not None:
        rotation_vectors, translation_vectors, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
        )

        for i, corner in enumerate(corners):
            qr_id = ids[i][0]
            qr_2d_coordinates = corner.reshape(4, 2).tolist()
            translation_vector = translation_vectors[i][0]
            rotation_vector = rotation_vectors[i][0]

            distance = np.linalg.norm(translation_vector)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            roll, pitch, yaw = calculate_yaw_pitch_roll(rotation_matrix)

            csv_output.append([frame_id, qr_id, qr_2d_coordinates, distance, yaw, pitch, roll])

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    return frame, csv_output

def process_video(video_path: str, output_video_path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """Process the input video to detect ArUco markers and save the detection data."""
    capture = initialize_video_capture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = initialize_video_writer(output_video_path, fps, frame_width, frame_height)
    aruco_dict, aruco_parameters = initialize_aruco_detector()

    csv_output = []
    csv_headers = ['Frame_ID', 'QR_ID', 'QR_2D_Coordinates', 'Distance', 'Yaw', 'Pitch', 'Roll']
    frame_id = 0

    while capture.isOpened():
        start_time = time.time()

        ret, frame = capture.read()
        if not ret:
            break

        processed_frame, frame_csv_output = process_frame(frame, aruco_dict, aruco_parameters, camera_matrix, dist_coeffs, frame_id)
        csv_output.extend(frame_csv_output)
        video_writer.write(processed_frame)
        frame_id += 1

        elapsed_time = time.time() - start_time
        if elapsed_time < 0.03:
            time.sleep(0.03 - elapsed_time)

    capture.release()
    video_writer.release()

    df = pd.DataFrame(csv_output, columns=csv_headers)
    df.to_csv(output_video_path.replace('.mp4', '_detected_markers.csv'), index=False)

def main():
    """Main function to initialize parameters and process the video."""
    video_path = "C:\\Users\\chai9\\OneDrive\\שולחן העבודה\\QR\\challengeB.mp4"
    output_video_path = 'output_video.mp4'

    camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                              [0.000000, 919.018377, 351.238301],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    process_video(video_path, output_video_path, camera_matrix, dist_coeffs)

if __name__ == "__main__":
    main()
