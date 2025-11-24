import csv
import os
import time


def save_measurements(csv_file, measurement_id, user_profile, measurements_cm, measurements_inch, skin_tone):
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0

    with open(csv_file, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if not file_exists:
            writer.writerow([
                "Measurement_ID",
                "Name",
                "Age",
                "Timestamp",
                "Height_cm",
                "Height_inch",
                "Shoulder_cm",
                "Shoulder_inch",
                "Chest_cm",
                "Chest_inch",
                "Waist_cm",
                "Waist_inch",
                "Hip_cm",
                "Hip_inch",
                "Neck_cm",
                "Neck_inch",
                "Arm_Length_cm",
                "Arm_Length_inch",
                "Leg_Length_cm",
                "Leg_Length_inch",
                "Inseam_cm",
                "Inseam_inch",
                "Body_Depth_cm",
                "Body_Depth_inch",
                "Skin_Tone",
            ])

        writer.writerow([
            measurement_id,
            user_profile.name,
            user_profile.age,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            measurements_cm['Height'],
            measurements_inch['Height'],
            measurements_cm['Shoulder'],
            measurements_inch['Shoulder'],
            measurements_cm['Chest'],
            measurements_inch['Chest'],
            measurements_cm['Waist'],
            measurements_inch['Waist'],
            measurements_cm['Hip'],
            measurements_inch['Hip'],
            measurements_cm['Neck'],
            measurements_inch['Neck'],
            measurements_cm['Arm_Length'],
            measurements_inch['Arm_Length'],
            measurements_cm['Leg_Length'],
            measurements_inch['Leg_Length'],
            measurements_cm['Inseam'],
            measurements_inch['Inseam'],
            measurements_cm['Body_Depth'],
            measurements_inch['Body_Depth'],
            skin_tone,
        ])
