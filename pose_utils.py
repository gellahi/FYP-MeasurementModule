import cv2
import numpy as np


def auto_rotate(frame):
    height, width = frame.shape[:2]
    if width < height:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def calculate_distance(point_one, point_two, frame_shape):
    """Calculate pixel distance between two landmarks."""
    height, width = frame_shape[:2]
    x_diff = (point_one.x - point_two.x) * width
    y_diff = (point_one.y - point_two.y) * height
    return np.sqrt(x_diff * 2 + y_diff * 2)


def check_pose_stability(landmarks_history, threshold=5):
    """Check if pose is stable across frames."""
    if len(landmarks_history) < 2:
        return False

    last = landmarks_history[-1]
    prev = landmarks_history[-2]

    total_movement = 0
    for i in range(len(last)):
        movement = np.sqrt((last[i].x - prev[i].x) * 2 + (last[i].y - prev[i].y) * 2)
        total_movement += movement

    return total_movement * 100 < threshold


def check_full_body_visible(landmarks):
    """Verify all key points are visible."""
    key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    visible_count = sum(1 for i in key_points if landmarks[i].visibility > 0.7)
    return visible_count >= 11


def check_front_pose(landmarks):
    """Check if person is facing camera (front pose)."""
    left_shoulder_vis = landmarks[11].visibility
    right_shoulder_vis = landmarks[12].visibility
    left_hip_vis = landmarks[23].visibility
    right_hip_vis = landmarks[24].visibility

    if min(left_shoulder_vis, right_shoulder_vis, left_hip_vis, right_hip_vis) < 0.7:
        return False

    shoulder_y_diff = abs(landmarks[11].y - landmarks[12].y)
    return shoulder_y_diff < 0.05


def check_side_pose(landmarks):
    """Check if person is in side pose - strict version to avoid front pose."""
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    nose = landmarks[0]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    shoulder_width_x = abs(left_shoulder.x - right_shoulder.x)
    hip_width_x = abs(left_hip.x - right_hip.x)

    shoulders_narrow = shoulder_width_x < 0.10
    hips_narrow = hip_width_x < 0.10

    nose_shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
    nose_alignment = abs(nose.x - nose_shoulder_avg_x) < 0.15

    return shoulders_narrow and hips_narrow


def draw_ui(frame, status, color, measurements=None):
    """Draw consistent UI elements."""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, status, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

    if measurements:
        y_pos = 110
        for key, value in measurements.items():
            text = f"{key}: {value} cm"
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y_pos += 40
