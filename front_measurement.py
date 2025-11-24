from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from pose_utils import (
    auto_rotate,
    calculate_distance,
    check_full_body_visible,
    check_front_pose,
    check_pose_stability,
    draw_ui,
)


def capture_front_pose_measurements(cap, pose, px_to_cm, calibration_frames, measurement_samples):
    print("\n" + "=" * 50)
    print("FRONT POSE MEASUREMENT")
    print("=" * 50)
    print("\nPlease face the camera directly and remain stable.")

    landmarks_history = deque(maxlen=calibration_frames)
    shoulder_samples = []
    arm_length_samples = []
    leg_length_samples = []

    stable_frames = 0
    mp_draw = mp.solutions.drawing_utils

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            continue

        frame = auto_rotate(raw_frame)
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Position full body in frame"
        color = (0, 0, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmarks_history.append(landmarks)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            if check_full_body_visible(landmarks) and check_front_pose(landmarks):
                if check_pose_stability(landmarks_history):
                    stable_frames += 1
                    color = (0, 255, 0)
                    status = f"CAPTURING ({stable_frames}/{calibration_frames}) - Hold position"

                    if stable_frames >= calibration_frames:
                        shoulder_px = calculate_distance(landmarks[11], landmarks[12], frame.shape)
                        shoulder_cm = shoulder_px * px_to_cm
                        shoulder_samples.append(shoulder_cm)

                        left_arm_px = calculate_distance(landmarks[11], landmarks[15], frame.shape)
                        right_arm_px = calculate_distance(landmarks[12], landmarks[16], frame.shape)
                        arm_length_cm = ((left_arm_px + right_arm_px) / 2) * px_to_cm
                        arm_length_samples.append(arm_length_cm)

                        left_leg_px = calculate_distance(landmarks[23], landmarks[27], frame.shape)
                        right_leg_px = calculate_distance(landmarks[24], landmarks[28], frame.shape)
                        leg_length_cm = ((left_leg_px + right_leg_px) / 2) * px_to_cm
                        leg_length_samples.append(leg_length_cm)

                        if len(shoulder_samples) >= measurement_samples:
                            break
                else:
                    stable_frames = 0
                    status = "Please remain stable"
                    color = (0, 165, 255)
            else:
                stable_frames = 0
                if not check_full_body_visible(landmarks):
                    status = "Full body not visible"
                elif not check_front_pose(landmarks):
                    status = "Please face the camera directly"

        draw_ui(frame, status, color)
        cv2.imshow("Front Pose Measurement", frame)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    shoulder_cm = round(np.median(shoulder_samples), 1)
    arm_length_cm = round(np.median(arm_length_samples), 1)
    leg_length_cm = round(np.median(leg_length_samples), 1)

    print(f"\n\u2713 Front pose measurements captured successfully!")
    print(f"  Shoulder width: {shoulder_cm} cm")
    print(f"  Arm length: {arm_length_cm} cm")
    print(f"  Leg length: {leg_length_cm} cm")

    cv2.destroyWindow("Front Pose Measurement")

    return shoulder_cm, arm_length_cm, leg_length_cm
