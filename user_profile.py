import csv
import os
import time


class UserProfile:
    """Handles user data input and storage."""

    PROFILE_FILENAME = 'user_profile.csv'

    def _init_(self):
        self.name = None
        self.age = None
        self.skin_tone = "N/A"
        self.size_summary = "N/A"
        self.last_measurement_id = f"{time.time():.0f}"

    def input_initial_data(self):
        """Asks the user for name and age."""
        print("\n--- User Profile Setup ---")
        while True:
            name = input("Enter your Name: ").strip()
            if name:
                self.name = name
                break
            print("Name cannot be empty.")

        while True:
            try:
                age = int(input("Enter your Age: ").strip())
                if 5 <= age <= 100:
                    self.age = age
                    break
                print("Please enter a realistic age.")
            except ValueError:
                print("Invalid input. Please enter a number for age.")

        print(f"\n[PROFILE] Profile created for {self.name} (Age: {self.age}).")

    def set_detected_data(self, skin_tone, measurements, measurement_id):
        """Sets the detected dominant skin tone, generates a size summary, and updates ID."""
        self.skin_tone = skin_tone
        self.last_measurement_id = measurement_id
        self._generate_size_summary(measurements)

    def _generate_size_summary(self, measurements):
        """Generates a simple size summary including all primary measurements."""
        if not measurements or not any(v > 0 for v in measurements.values()):
            self.size_summary = "No valid measurements recorded."
            return

        summary_parts = []

        chest = measurements.get('Chest', 0)
        waist = measurements.get('Waist', 0)
        shoulder = measurements.get('Shoulder', 0)

        if chest > 110 or shoulder > 48:
            size = "L/XL"
        elif chest > 90 or shoulder > 42:
            size = "M"
        else:
            size = "S"

        summary_parts.append(f"Size: {size}")
        summary_parts.append(f"C: {chest:.1f} cm")
        summary_parts.append(f"W: {waist:.1f} cm")
        summary_parts.append(f"Sh: {shoulder:.1f} cm")

        self.size_summary = ", ".join(summary_parts)

    def save_profile_to_csv(self):
        """Saves or updates the user profile to a SEPARATE CSV."""
        file_exists = os.path.isfile(self.PROFILE_FILENAME)

        data = {
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Name': self.name,
            'Age': self.age,
            'Last_Measurement_ID': self.last_measurement_id,
            'Detected_Skin_Tone': self.skin_tone,
            'Size_Summary': self.size_summary
        }

        try:
            with open(self.PROFILE_FILENAME, mode='a', newline='') as file:
                fieldnames = list(data.keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(data)

            print(f"\n[SUCCESS] Profile saved to '{self.PROFILE_FILENAME}'")

        except Exception as e:
            print(f"\n[ERROR] Failed to save profile CSV: {e}")
