import cv2
import mediapipe as mp
import time

from calibration import run_height_calibration
from config import CALIBRATION_FRAMES, CSV_FILE, MEASUREMENT_SAMPLES
from data_storage import save_measurements
from final_calculations import compute_final_measurements
from front_measurement import capture_front_pose_measurements
from side_measurement import capture_side_pose_depth
from summary_display import display_measurement_summary
from user_profile import UserProfile


def setup_pose_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        smooth_landmarks=True,
    )
    return pose


def setup_camera():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    print("\n" + "=" * 60)
    print("    ADVANCED BODY MEASUREMENT SYSTEM")
    print("=" * 60)

    user_profile = UserProfile()
    user_profile.input_initial_data()
    measurement_id = f"{user_profile.name}_{time.time():.0f}"

    pose = setup_pose_model()
    cap = setup_camera()

    print("\n" + "=" * 50)
    print("HEIGHT-BASED CALIBRATION")
    print("=" * 50)
    print("\nInstructions:")
    print("1. Stand upright facing the camera")
    print("2. Ensure your full body is visible (head to feet)")
    print("3. Keep the camera stable (tripod recommended)")
    print("4. Stand 1.5-2 meters away from the camera")

    known_height_cm = float(input("\nPlease enter your exact height (cm): "))

    print(f"\n\u2713 Height entered: {known_height_cm} cm")
    print("\nCapturing full body calibration data...")
    print("Please position yourself in frame and remain stable...")

    px_to_cm, detected_skin_tone = run_height_calibration(
        cap,
        pose,
        known_height_cm,
        CALIBRATION_FRAMES,
    )

    shoulder_cm, arm_length_cm, leg_length_cm = capture_front_pose_measurements(
        cap,
        pose,
        px_to_cm,
        CALIBRATION_FRAMES,
        MEASUREMENT_SAMPLES,
    )

    depth_cm = capture_side_pose_depth(
        cap,
        pose,
        px_to_cm,
        shoulder_cm,
        MEASUREMENT_SAMPLES,
    )

    measurements_cm, measurements_inch = compute_final_measurements(
        known_height_cm,
        shoulder_cm,
        arm_length_cm,
        leg_length_cm,
        depth_cm,
    )

    save_measurements(
        CSV_FILE,
        measurement_id,
        user_profile,
        measurements_cm,
        measurements_inch,
        detected_skin_tone,
    )

    user_profile.set_detected_data(detected_skin_tone, measurements_cm, measurement_id)
    user_profile.save_profile_to_csv()

    display_measurement_summary(user_profile, measurements_cm, measurements_inch, detected_skin_tone)

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("\u2713 MEASUREMENT PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nData saved to:")
    print(f"  • Body measurements: {CSV_FILE}")
    print(f"  • User profile: {UserProfile.PROFILE_FILENAME}")
    print(f"  • Measurement ID: {measurement_id}")
    print("\n" + "=" * 60)
    print("System ready for virtual try-on integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()