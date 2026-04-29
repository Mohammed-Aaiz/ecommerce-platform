import cv2
import numpy as np
import mediapipe as mp
from rembg import remove, new_session
import os
import math


def try_on(person_path, garment_path, output_path="outputs/result.png"):

    os.makedirs("outputs", exist_ok=True)

    # Load images
    person = cv2.imread(person_path)
    garment = cv2.imread(garment_path)

    # Background removal
    session = new_session("u2net")
    person = np.array(remove(person, session=session))
    garment = np.array(remove(garment, session=session))

    if person.shape[2] == 3:
        person = cv2.cvtColor(person, cv2.COLOR_BGR2BGRA)

    if garment.shape[2] == 3:
        garment = cv2.cvtColor(garment, cv2.COLOR_BGR2BGRA)

    # Pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    rgb = cv2.cvtColor(person[:, :, :3], cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    h, w, _ = person.shape
    lm = result.pose_landmarks.landmark

    # Key points
    ls = lm[11]
    rs = lm[12]
    lh = lm[23]
    rh = lm[24]

    x1, y1 = int(ls.x * w), int(ls.y * h)
    x2, y2 = int(rs.x * w), int(rs.y * h)
    hx1, hy1 = int(lh.x * w), int(lh.y * h)
    hx2, hy2 = int(rh.x * w), int(rh.y * h)

    # Torso region
    top_y = int((y1 + y2) / 2)
    bottom_y = int((hy1 + hy2) / 2)

    torso_height = bottom_y - top_y
    center_x = int((x1 + x2) / 2)

    # --- ROTATION ---
    angle = -math.degrees(math.atan2(y2 - y1, x2 - x1))

    # --- SCALE ---
    garment_height = int(torso_height * 1.3)
    aspect_ratio = garment.shape[1] / garment.shape[0]
    garment_width = int(garment_height * aspect_ratio)

    garment_resized = cv2.resize(garment, (garment_width, garment_height))

    # --- ROTATE ---
    (hg, wg) = garment_resized.shape[:2]
    center = (wg // 2, hg // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    garment_rotated = cv2.warpAffine(
        garment_resized,
        M,
        (wg, hg),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # --- POSITION ---
    x_offset = int(center_x - garment_width / 2)
    y_offset = int(top_y + 10)

    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    # --- OVERLAY ---
    for y in range(garment_rotated.shape[0]):
        for x in range(garment_rotated.shape[1]):

            if y + y_offset >= h or x + x_offset >= w:
                continue

            alpha = garment_rotated[y, x][3] / 255.0

            for c in range(3):
                person[y + y_offset, x + x_offset][c] = (
                    alpha * garment_rotated[y, x][c] +
                    (1 - alpha) * person[y + y_offset, x + x_offset][c]
                )

    result_img = cv2.cvtColor(person, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(output_path, result_img)

    return output_path