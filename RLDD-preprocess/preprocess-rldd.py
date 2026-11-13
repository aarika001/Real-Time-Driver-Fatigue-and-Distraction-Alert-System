import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# CONSTANTS & LANDMARK DEFINITIONS
# ==============================================================================
TARGET_FPS = 15
EYE_CROP_SIZE = (64, 64)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi")

# Label mapping: KSS filename -> Unified System Label
LABEL_MAP = {
    "0": "alert",
    "5": "low_vigilant",
    "10": "fatigue"
}

MOUTH_IDXS = [61, 291, 39, 181, 0, 17, 269, 405]
LEFT_EYE_IDXS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDXS = [362, 263, 387, 386, 385, 373, 374, 380]

FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),     # Left eye left corner (33)
    (225.0, 170.0, -135.0),      # Right eye right corner (263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (61)
    (150.0, -150.0, -125.0)      # Right mouth corner (291)
], dtype=np.float64)
POSE_2D_IDXS = [1, 152, 33, 263, 61, 291]


# ==============================================================================
# FEATURE & QUALITY EXTRACTION HELPERS
# ==============================================================================
def compute_mar(landmarks, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in MOUTH_IDXS])
    v1 = np.linalg.norm(pts[2] - pts[3])
    v2 = np.linalg.norm(pts[4] - pts[5])
    v3 = np.linalg.norm(pts[6] - pts[7])
    horiz = np.linalg.norm(pts[0] - pts[1])
    return (v1 + v2 + v3) / (3.0 * horiz + 1e-6)


def compute_head_pose(landmarks, w, h):
    pts_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_2D_IDXS], dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2.0],
        [0, focal_length, h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(FACE_3D_MODEL, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def extract_eye_crop_and_quality(gray_frame, landmarks, eye_idxs, w, h, pad_ratio=0.35):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_idxs])
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half_size = max(box_w, box_h) * (0.5 + pad_ratio)

    x1 = max(0, int(cx - half_size))
    y1 = max(0, int(cy - half_size))
    x2 = min(w, int(cx + half_size))
    y2 = min(h, int(cy + half_size))

    crop = gray_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(EYE_CROP_SIZE, dtype=np.uint8), [0.0, 0.0, 1.0, 0.0]

    brightness = float(np.mean(crop) / 255.0)
    contrast = float(np.std(crop) / 255.0)
    glare_frac = float(np.mean(crop >= 245))
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())

    resized = cv2.resize(crop, EYE_CROP_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    norm_crop = clahe.apply(resized)

    return norm_crop, [brightness, contrast, glare_frac, sharpness]


# ==============================================================================
# DISCOVERY & MAIN PIPELINE
# ==============================================================================
def discover_videos(rldd_dir, include_low_vigilant=True):
    records = []
    for dirpath, _, filenames in os.walk(rldd_dir):
        for fname in filenames:
            if not fname.lower().endswith(VIDEO_EXTENSIONS):
                continue

            stem = os.path.splitext(fname)[0]
            if stem not in LABEL_MAP:
                continue
            if stem == "5" and not include_low_vigilant:
                continue

            full_path = os.path.join(dirpath, fname)
            parts = os.path.normpath(full_path).split(os.sep)

            subject_id = parts[-2]
            fold_id = next((p.split("_")[0] for p in parts if p.startswith("Fold")), "Unknown")

            records.append({
                "fold_id": fold_id,
                "subject_id": subject_id,
                "raw_kss": stem,
                "label": LABEL_MAP[stem],
                "video_path": full_path
            })
    return pd.DataFrame(records)


def process_video(video_path, landmarker, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0

    frame_interval = max(1, round(orig_fps / target_fps))

    mar_seq, pose_seq, quality_seq, detected_seq = [], [], [], []
    left_crops, right_crops = [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                lms = result.face_landmarks[0]
                mar = compute_mar(lms, w, h)
                pitch, yaw, roll = compute_head_pose(lms, w, h)

                l_crop, l_qual = extract_eye_crop_and_quality(gray_frame, lms, LEFT_EYE_IDXS, w, h)
                r_crop, r_qual = extract_eye_crop_and_quality(gray_frame, lms, RIGHT_EYE_IDXS, w, h)
                avg_qual = [(l + r) / 2.0 for l, r in zip(l_qual, r_qual)]

                mar_seq.append(mar)
                pose_seq.append([pitch, yaw, roll])
                quality_seq.append(avg_qual)
                left_crops.append(l_crop)
                right_crops.append(r_crop)
                detected_seq.append(1)
            else:
                mar_seq.append(0.0)
                pose_seq.append([0.0, 0.0, 0.0])
                quality_seq.append([0.0, 0.0, 1.0, 0.0])
                left_crops.append(np.zeros(EYE_CROP_SIZE, dtype=np.uint8))
                right_crops.append(np.zeros(EYE_CROP_SIZE, dtype=np.uint8))
                detected_seq.append(0)

        frame_idx += 1

    cap.release()
    if len(detected_seq) == 0:
        return None

    return {
        "mar": np.array(mar_seq, dtype=np.float32),
        "head_pose": np.array(pose_seq, dtype=np.float32),
        "quality": np.array(quality_seq, dtype=np.float32),
        "face_detected": np.array(detected_seq, dtype=np.uint8),
        "left_eye_crops": np.array(left_crops, dtype=np.uint8),
        "right_eye_crops": np.array(right_crops, dtype=np.uint8)
    }


def main():
    parser = argparse.ArgumentParser(description="UTA-RLDD Preprocessing Pipeline (Role R3)")
    parser.add_argument("--rldd_dir", type=str, required=True, help="Path to root RLDD folder containing Fold1_part1, Fold3_part2, etc.")
    parser.add_argument("--output_dir", type=str, default="./processed_rldd", help="Directory to save processed .npz files and metadata CSV.")
    parser.add_argument("--model_path", type=str, default="./face_landmarker.task", help="Path to MediaPipe face_landmarker.task file.")
    parser.add_argument("--target_fps", type=int, default=TARGET_FPS, help="Target sampling frame rate (default: 15).")
    parser.add_argument("--exclude_low_vigilant", action="store_true", help="Exclude KSS 5 (5.mp4) videos from processing.")
    args = parser.parse_args()

    if not os.path.isdir(args.rldd_dir):
        raise FileNotFoundError(f"RLDD directory not found: {args.rldd_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"MediaPipe model not found: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = discover_videos(args.rldd_dir, include_low_vigilant=not args.exclude_low_vigilant)
    print(f"Discovered {len(df)} videos across {df['subject_id'].nunique()} subjects in '{args.rldd_dir}'.")

    base_options = python.BaseOptions(model_asset_path=args.model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    processed_records = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing UTA-RLDD"):
            data = process_video(row["video_path"], landmarker, args.target_fps)
            if data is None:
                continue

            out_subdir = os.path.join(args.output_dir, row["fold_id"], f"subject_{row['subject_id']}")
            os.makedirs(out_subdir, exist_ok=True)
            out_file = os.path.join(out_subdir, f"kss_{row['raw_kss']}_{row['label']}.npz")

            np.savez_compressed(out_file, **data)

            rec = row.to_dict()
            rec["npz_path"] = out_file
            rec["num_frames_15fps"] = len(data["face_detected"])
            rec["face_detection_rate"] = float(np.mean(data["face_detected"]))
            processed_records.append(rec)

    meta_df = pd.DataFrame(processed_records)
    meta_csv_path = os.path.join(args.output_dir, "rldd_metadata_index.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"Processing complete. Master metadata index saved to: {meta_csv_path}")


if __name__ == "__main__":
    main()