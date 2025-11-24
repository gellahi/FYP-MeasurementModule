import cv2
import mediapipe as mp
import numpy as np

FACE_SAMPLE_POINTS = [
    mp.solutions.pose.PoseLandmark.NOSE,
    mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
    mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER
]


def get_skin_tone_name(h, s, v):
    """Maps averaged HSV values (0-255 range) to a descriptive skin tone name."""
    v_norm = v * 100 / 255
    s_norm = s * 100 / 255

    if v_norm < 30:
        return "Deep Brown/Black"
    if v_norm < 50:
        return "Dark" if s_norm < 25 else "Medium-Dark"
    if v_norm < 70:
        return "Medium" if s_norm < 20 else "Medium-Tan"
    if v_norm < 90:
        return "Light" if s_norm < 15 else "Fair/Pale"

    return "Very Light/White"


def get_person_skin_tone(image, landmarks, width, height):
    """Analyzes specific facial pixel areas for a robust skin tone reading."""
    if not landmarks or not landmarks.landmark:
        return "N/A"

    sample_colors = []

    for lm_enum in FACE_SAMPLE_POINTS:
        lm = landmarks.landmark[lm_enum]

        if lm.visibility > 0.8:
            px_x = int(lm.x * width)
            px_y = int(lm.y * height)

            sample_size = 5
            x_min = max(0, px_x - sample_size)
            x_max = min(width, px_x + sample_size)
            y_min = max(0, px_y - sample_size)
            y_max = min(height, px_y + sample_size)

            if x_max > x_min and y_max > y_min:
                sample_area = image[y_min:y_max, x_min:x_max]
                hsv_sample = cv2.cvtColor(sample_area, cv2.COLOR_BGR2HSV)

                avg_h, avg_s, avg_v = np.mean(hsv_sample, axis=(0, 1))
                sample_colors.append((avg_h, avg_s, avg_v))

    if not sample_colors:
        return "N/A"

    avg_h, avg_s, avg_v = np.mean(sample_colors, axis=0)
    return get_skin_tone_name(avg_h, avg_s, avg_v)
