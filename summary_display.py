import cv2
import numpy as np


def display_measurement_summary(user_profile, measurements_cm, measurements_inch, skin_tone):
    result_img = np.zeros((950, 1400, 3), np.uint8)
    result_img[:] = (20, 30, 60)

    cv2.putText(
        result_img,
        "MEASUREMENT SUMMARY - COMPLETE",
        (180, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        2,
        (0, 255, 255),
        5,
    )

    cv2.putText(
        result_img,
        f"Name: {user_profile.name}",
        (80, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 200, 100),
        2,
    )
    cv2.putText(
        result_img,
        f"Age: {user_profile.age} | Skin Tone: {skin_tone}",
        (80, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 200, 100),
        2,
    )
    cv2.putText(
        result_img,
        f"Size Classification: {user_profile.size_summary}",
        (80, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (100, 255, 100),
        2,
    )

    cv2.line(result_img, (50, 220), (1350, 220), (100, 100, 100), 2)

    cv2.putText(result_img, "Measurement", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    cv2.putText(result_img, "CM", (600, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    cv2.putText(result_img, "Inches", (900, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

    cv2.line(result_img, (50, 280), (1350, 280), (100, 100, 100), 2)

    measurements = [
        ("Height", 'Height'),
        ("Shoulder Width", 'Shoulder'),
        ("Chest", 'Chest'),
        ("Waist", 'Waist'),
        ("Hip", 'Hip'),
        ("Neck", 'Neck'),
        ("Arm Length", 'Arm_Length'),
        ("Leg Length", 'Leg_Length'),
        ("Inseam", 'Inseam'),
        ("Body Depth", 'Body_Depth'),
    ]

    y_pos = 330
    for label, key in measurements:
        cv2.putText(result_img, label, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(result_img, f"{measurements_cm[key]:.1f}", (600, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(result_img, f"{measurements_inch[key]:.1f}", (900, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        y_pos += 60

    cv2.putText(result_img, "Press any key to exit", (450, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    cv2.imshow("Measurement Summary", result_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Measurement Summary")
