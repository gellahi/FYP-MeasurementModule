import cv2
import mediapipe as mp
import numpy as np

from pose_utils import auto_rotate, check_side_pose


def capture_side_pose_depth(cap, pose, px_to_cm, shoulder_cm, measurement_samples):
    print("\n" + "=" * 50)
    print("SIDE POSE MEASUREMENT")
    print("=" * 50)
    print("\nInstructions:")
    print("- Turn 90 degrees to your left or right")
    print("- Stand in perfect profile view")
    print("- Ensure shoulders are aligned (overlapping)")
    print("- Keep full body visible in frame")
    print("- Press ESC to skip if needed")

    input("\nPress ENTER when ready...")

    mp_draw = mp.solutions.drawing_utils
    landmarks_history = []
    depth_samples = []
    stable_frames = 0
    side_timeout = 0
    max_side_attempts = 600
    min_stable_frames = 20

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            continue

        side_timeout += 1

        if side_timeout > max_side_attempts:
            print("\n\u26A0 Timeout - Using estimated depth measurement")
            if len(depth_samples) == 0:
                depth_samples = [shoulder_cm * 0.42]
            break

        frame = auto_rotate(raw_frame)
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        status = "Turn 90 degrees sideways"
        color = (0, 0, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmarks_history.append(landmarks)
            if len(landmarks_history) > 3:
                landmarks_history.pop(0)

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            shoulder_width_x = abs(landmarks[11].x - landmarks[12].x)
            hip_width_x = abs(landmarks[23].x - landmarks[24].x)

            is_front = shoulder_width_x > 0.12 or hip_width_x > 0.12

            cv2.putText(
                frame,
                f"Shoulder alignment: {shoulder_width_x:.3f} (target <0.10)",
                (20, height - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Hip alignment: {hip_width_x:.3f} (target <0.10)",
                (20, height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if is_front:
                status = "FRONT POSE DETECTED - Turn 90 degrees sideways"
                color = (0, 0, 255)
                stable_frames = 0
                cv2.putText(
                    frame,
                    ">>> TURN MORE SIDEWAYS <<<",
                    (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.5,
                    (0, 0, 255),
                    4,
                )
            else:
                is_side = check_side_pose(landmarks)

                if is_side:
                    if len(landmarks_history) >= 3:
                        stable_frames += 1
                        color = (0, 255, 0)
                        status = f"EXCELLENT! Hold position ({stable_frames}/{min_stable_frames})"

                        if stable_frames >= min_stable_frames:
                            depth_px = None

                            if landmarks[11].visibility > 0.4 and landmarks[23].visibility > 0.4:
                                depth_px = abs(landmarks[11].x - landmarks[23].x) * width
                            elif landmarks[12].visibility > 0.4 and landmarks[24].visibility > 0.4:
                                depth_px = abs(landmarks[12].x - landmarks[24].x) * width

                            if depth_px is None or depth_px < 15:
                                shoulder_avg_x = (landmarks[11].x + landmarks[12].x) / 2
                                depth_px = abs(landmarks[0].x - shoulder_avg_x) * width

                            if depth_px and depth_px > 15:
                                depth_cm = depth_px * px_to_cm

                                if 10 < depth_cm < 60:
                                    depth_samples.append(depth_cm)
                                    print(f"\u2713 Sample {len(depth_samples)}: {depth_cm:.1f} cm")

                            if len(depth_samples) >= measurement_samples:
                                print(f"\n\u2713 Measurement complete! {len(depth_samples)} samples collected")
                                break
                else:
                    stable_frames = max(0, stable_frames - 1)
                    status = "Turn more sideways for profile view"
                    color = (0, 165, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 3)

        if stable_frames > 0:
            bar_width = int((stable_frames / min_stable_frames) * (width - 40))
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 120), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 90), (width - 20, 120), (100, 100, 100), 2)

        time_left = (max_side_attempts - side_timeout) // 30
        cv2.putText(
            frame,
            f"ESC-Skip | Samples:{len(depth_samples)}/{measurement_samples} | Time:{time_left}s",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Side Pose Measurement", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("\n\u26A0 Side measurement skipped by user")
            if len(depth_samples) == 0:
                depth_samples = [shoulder_cm * 0.42]
            break

    if len(depth_samples) == 0:
        depth_samples = [shoulder_cm * 0.42]
        print("\u26A0 Using estimated depth measurement")

    depth_cm = round(np.median(depth_samples), 1)
    print(f"\n\u2713 Body depth measurement: {depth_cm} cm (from {len(depth_samples)} samples)")

    cv2.destroyWindow("Side Pose Measurement")
    return depth_cm
