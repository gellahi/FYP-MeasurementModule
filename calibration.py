import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from pose_utils import auto_rotate, check_full_body_visible, check_pose_stability, draw_ui
from skin_tone import get_person_skin_tone


def run_height_calibration(cap, pose, known_height_cm, calibration_frames):
    height_samples = []
    stable_frames = 0
    calibration_done = False
    landmarks_history_cal = deque(maxlen=calibration_frames)
    mp_draw = mp.solutions.drawing_utils
    last_results = None
    last_frame = None
    frame_width = None
    frame_height = None

    while not calibration_done:
        ret, raw_frame = cap.read()
        if not ret:
            continue

        last_frame = auto_rotate(raw_frame)
        last_frame = cv2.flip(last_frame, 1)
        frame_height, frame_width = last_frame.shape[:2]
        rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        last_results = results

        status = "Position full body in frame (head to ankles)"
        color = (0, 0, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmarks_history_cal.append(landmarks)

            mp_draw.draw_landmarks(last_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            if check_full_body_visible(landmarks):
                nose_y = landmarks[0].y
                left_ankle_y = landmarks[27].y
                right_ankle_y = landmarks[28].y
                ankle_y = max(left_ankle_y, right_ankle_y)

                body_height_px = (ankle_y - nose_y) * frame_height

                if body_height_px > 100:
                    if check_pose_stability(landmarks_history_cal):
                        stable_frames += 1
                        color = (0, 255, 0)
                        status = f"EXCELLENT! Hold position ({stable_frames}/{calibration_frames})"

                        if stable_frames >= calibration_frames:
                            height_samples.append(body_height_px)

                            if len(height_samples) >= 5:
                                calibration_done = True
                    else:
                        stable_frames = 0
                        status = "Please remain stable"
                        color = (0, 165, 255)
                else:
                    status = "Please step back - full body required"
            else:
                stable_frames = 0
                status = "Full body not detected - adjust position"

        draw_ui(last_frame, status, color)
        cv2.imshow("Height Calibration", last_frame)

        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    reference_height_px = np.median(height_samples)
    px_to_cm = known_height_cm / reference_height_px

    print(f"\n\u2713 Calibration Complete!")
    print(f"  Height in pixels: {reference_height_px:.1f} px")
    print(f"  Conversion ratio: {px_to_cm:.4f} cm/px")
    print(f"  Estimated accuracy: ~95%")

    detected_skin_tone = "N/A"
    if last_results and last_results.pose_landmarks and last_frame is not None:
        detected_skin_tone = get_person_skin_tone(last_frame, last_results.pose_landmarks, frame_width, frame_height)
        print(f"  Detected skin tone: {detected_skin_tone}")

    cv2.destroyWindow("Height Calibration")
    time.sleep(1)

    return px_to_cm, detected_skin_tone
