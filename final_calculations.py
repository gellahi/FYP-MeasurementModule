def cm_to_inches(cm_value):
    return round(cm_value / 2.54, 1)


def compute_final_measurements(known_height_cm, shoulder_cm, arm_length_cm, leg_length_cm, depth_cm):
    print(f"\nApplying Final Accuracy Boost...")

    shoulder_cm_corrected = round(shoulder_cm * 1.09, 1)
    arm_length_cm = round(arm_length_cm * 1.10, 1)
    leg_length_cm = round(leg_length_cm * 1.02, 1)

    chest_cm = round(shoulder_cm_corrected * 2.18 + depth_cm * 0.88 + 3.2, 1)
    waist_cm = round(chest_cm * 0.855, 1)
    hip_cm = round(chest_cm * 1.02 + 2.5, 1)
    neck_cm = round(shoulder_cm_corrected * 0.92 + 1.8, 1)
    inseam_cm = round(leg_length_cm * 0.765, 1)

    if waist_cm > chest_cm:
        waist_cm = round(chest_cm * 0.92, 1)
    if hip_cm < waist_cm:
        hip_cm = round(waist_cm + 3, 1)

    measurement_cm = {
        'Height': known_height_cm,
        'Shoulder': shoulder_cm,
        'Chest': chest_cm,
        'Waist': waist_cm,
        'Hip': hip_cm,
        'Neck': neck_cm,
        'Arm_Length': arm_length_cm,
        'Leg_Length': leg_length_cm,
        'Inseam': inseam_cm,
        'Body_Depth': depth_cm,
    }

    measurement_inch = {key: cm_to_inches(value) for key, value in measurement_cm.items()}
    return measurement_cm, measurement_inch
