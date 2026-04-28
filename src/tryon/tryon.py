import cv2
import numpy as np
import mediapipe as mp
from rembg import remove


def try_on(person_path, garment_path, output_path="outputs/result.png"):

    # Load images
    person = cv2.imread(person_path)
    garment = cv2.imread(garment_path)

    # Remove background
    person = np.array(remove(person))
    garment = np.array(remove(garment))

    # ---------------- POSE DETECTION ----------------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        print("Pose not detected")
        return None

    h, w, _ = person.shape
    landmarks = result.pose_landmarks.landmark

    # Shoulders
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    x1, y1 = int(left_shoulder.x * w), int(left_shoulder.y * h)
    x2, y2 = int(right_shoulder.x * w), int(right_shoulder.y * h)

    # ---------------- SCALE GARMENT ----------------
    shoulder_width = abs(x2 - x1)

    garment_width = int(shoulder_width * 1.5)
    aspect_ratio = garment.shape[0] / garment.shape[1]
    garment_height = int(garment_width * aspect_ratio)

    garment_resized = cv2.resize(garment, (garment_width, garment_height))

    # ---------------- POSITION ----------------
    x_offset = min(x1, x2) - garment_width // 4
    y_offset = min(y1, y2)

    # Ensure inside bounds
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    # ---------------- OVERLAY ----------------
    for y in range(garment_resized.shape[0]):
        for x in range(garment_resized.shape[1]):

            if y + y_offset >= h or x + x_offset >= w:
                continue

            # Check alpha channel
            if garment_resized[y, x][3] > 0:
                person[y + y_offset, x + x_offset] = garment_resized[y, x][:3]

    # Save result
    cv2.imwrite(output_path, person)

    return output_path