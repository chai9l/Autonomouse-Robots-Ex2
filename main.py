import cv2
import numpy as np
import pandas as pd

# Load video
video_path = 'challengeB.mp4'
cap = cv2.VideoCapture(video_path)

# Define the aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Camera parameters - Taken from: https://tellopilots.com/threads/camera-intrinsic-parameter.2620/
camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

# FoV parameters (field of view in degrees)


def calculate_distance(tvec):
    return np.linalg.norm(tvec)

def calculate_yaw(rvec):
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Extract yaw angle
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.degrees(yaw)

def calculate_pose(corners):
    # This function uses solvePnP to get the rotation and translation vectors
    marker_length = 0.05  # Assuming a marker size of 5cm
    object_points = np.array([[-marker_length / 2, marker_length / 2, 0],
                              [marker_length / 2, marker_length / 2, 0],
                              [marker_length / 2, -marker_length / 2, 0],
                              [-marker_length / 2, -marker_length / 2, 0]])

    retval, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if retval:
        dist = calculate_distance(tvec)
        yaw = calculate_yaw(rvec)
        return dist, yaw
    else:
        return None, None


# DataFrame to store results
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            qr_id = ids[i][0]
            qr_corners = corners[i][0]
            dist, yaw = calculate_pose(qr_corners)
            if dist is not None:
                results.append({
                    'Frame ID': frame_id,
                    'QR id': qr_id,
                    'QR 2D': qr_corners.tolist(),
                    'Distance': dist,
                    'Yaw': yaw,
                })

            # Draw marker boundary and ID
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.putText(frame, str(qr_id), tuple(qr_corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame (for testing purposes)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(results)
df.to_csv('results/aruco_detection_results.csv', index=False)

output_video_path = 'results/output_with_aruco.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            qr_id = ids[i][0]
            qr_corners = corners[i][0]
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.putText(frame, str(qr_id), tuple(qr_corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()