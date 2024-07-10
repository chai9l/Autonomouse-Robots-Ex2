import math
import cv2
import csv
import numpy as np
from typing import Tuple


def initialize_video_capture(camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error")
        return None
    print("Webcam successfully connected.")
    return cap


def find_destination(csv_data, target_destination):
    for row in csv_data:
        if row[0] == target_destination:
            return row


def initialize_aruco_detector() -> Tuple[cv2.aruco_Dictionary, cv2.aruco_DetectorParameters]:
    """Initialize ArUco dictionary and detector parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_parameters = cv2.aruco.DetectorParameters()
    return aruco_dict, aruco_parameters


def detect_markers(frame, aruco_dict, parameters):
    return cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)


def estimate_pose(corners, marker_length, camera_matrix, dist_coeffs):
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    return rvecs[0], tvecs[0]


def draw_marker_and_info(frame, corners, camera_matrix, dist_coeffs, rvec, tvec, current_pose, current_distance,
                         command):
    cv2.aruco.drawDetectedMarkers(frame, corners)
    try:
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
    except AttributeError:
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

    yaw, pitch, roll = current_pose
    cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Distance: {current_distance:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame, f'Roll: {roll:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
    cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)

    if command:
        cv2.putText(frame, f'Command: {command[0]}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        print("Movement Command:", command[0])
    else:
        cv2.putText(frame, 'No Id detected go backward', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)


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


def load_csv(csv_file):
    csv_data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            frame_id = int(row[0])
            qr_id = int(row[1])
            qr_2d = eval(row[2])
            distance = float(row[3])
            yaw = float(row[4])
            pitch = float(row[5])
            roll = float(row[6])
            target_pose = (distance, yaw, pitch, roll)
            csv_data.append((frame_id, qr_id, qr_2d, target_pose))
    return csv_data

def calculate_midpoint(vertices):
    """
    Calculate the midpoint of a set of vertices.

    Args:
    vertices (List[Tuple[float, float]]): A list of vertex coordinates.

    Returns:
    Tuple[float, float]: The coordinates of the midpoint.

    the function assesses how far off the current position is from the target position in both x and y coordinates and adjust the movement accordingly.
    """
    sum_x_coord, sum_y_coord = 0, 0
    total_vertices = len(vertices)

    for vertex_x, vertex_y in vertices:
        sum_x_coord += vertex_x
        sum_y_coord += vertex_y

    midpoint_x = sum_x_coord / total_vertices
    midpoint_y = sum_y_coord / total_vertices

    return midpoint_x, midpoint_y

def generate_navigation_commands(current_position, target_position, current_distance, desired_distance, live_feed_data,
                                 target_vertices):
    """
    Generate navigation commands to move from the current position to the target position.

    Args:
    current_position (Tuple[float, float]): The current position coordinates.
    target_position (Tuple[float, float]): The target position coordinates.
    current_distance (float): The current distance to the target.
    desired_distance (float): The desired distance from the target.
    live_feed_data (np.ndarray): The live feed data containing positional information.
    target_vertices (np.ndarray): The target vertices data.

    Returns:
    List[str]: A list of navigation commands.
    """

    distance_difference = desired_distance - current_distance
    angle_difference = (target_position[1] - current_position[0]) % 180

    live_midpoint_x, live_midpoint_y = calculate_midpoint(live_feed_data[0][0].tolist())
    target_midpoint_x, target_midpoint_y = calculate_midpoint(target_vertices)
    x_offset = live_midpoint_x - target_midpoint_x
    y_offset = live_midpoint_y - target_midpoint_y

    navigation_commands = []

    if abs(distance_difference) > 1.5:
        navigation_commands.append("move-backward" if distance_difference > 0 else "move-forward")

    if abs(y_offset) > 15:
        navigation_commands.append("pitch-down" if y_offset > 0 else "pitch-up")

    if abs(x_offset) > 15:
        navigation_commands.append("yaw-right" if x_offset > 0 else "yaw-left")

    if abs(angle_difference) > 18:
        navigation_commands.append("rotate-right" if angle_difference > 0 else "rotate-left")

    navigation_commands.append("At Position")

    return navigation_commands if navigation_commands else ["At Position"]


def process_live_video(csv_data, target_frame_id, marker_length, camera_matrix, dist_coeffs):
    cap = initialize_video_capture(1)
    if cap is None:
        return

    target_args = find_destination(csv_data, target_frame_id)

    if not target_args:
        return

    destination_id = target_args[1]
    destination_corners = target_args[2]
    target_pose = target_args[3]
    destination_distance = target_pose[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        aruco_dict, parameters = initialize_aruco_detector()

        corners, ids, rejected = detect_markers(frame, aruco_dict, parameters)
        if ids is not None:
            for i in range(len(ids)):
                id = ids[i][0]
                if id == destination_id:
                    rvec, tvec = estimate_pose(corners[i], marker_length, camera_matrix, dist_coeffs)
                    curr_distance = np.linalg.norm(tvec)
                    R, _ = cv2.Rodrigues(rvec)
                    current_pose = calculate_yaw_pitch_roll(R)
                    command = generate_navigation_commands(current_pose, target_pose, curr_distance, destination_distance,
                                                           corners, destination_corners)

                    draw_marker_and_info(frame, corners, camera_matrix, dist_coeffs, rvec, tvec, current_pose,
                                          curr_distance, command)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                              [0.000000, 919.018377, 351.238301],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

    marker_length = 0.14

    csv_file = 'C:\\Users\\chai9\\OneDrive\\שולחן העבודה\\QR\\output_video_detected_markers.csv'
    csv_data = load_csv(csv_file)

    target_pos = int(input("Enter the target pos from the csv file: "))

    process_live_video(csv_data, target_pos, marker_length, camera_matrix, dist_coeffs)
