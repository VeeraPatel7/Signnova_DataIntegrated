import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
GESTURE_DATA_DIR = REPO_ROOT / "data" / "gesture_data"
ISL_CSLTR_DIR = REPO_ROOT / "data" / "ISL-CSLTR" / "Videos_Sentence_Level"

def extract_landmarks_from_frame(result):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Mirrored coordinate convention to match pipeline
            if label == "Left":
                right_hand = landmarks
            else:
                left_hand = landmarks

    return left_hand + right_hand

def process_isl_csltr():
    if not ISL_CSLTR_DIR.exists():
        raise FileNotFoundError(f"Directory not found at {ISL_CSLTR_DIR}")

    folders = [f for f in os.listdir(ISL_CSLTR_DIR) if os.path.isdir(ISL_CSLTR_DIR / f)]
    print(f"Found {len(folders)} sentence classes in {ISL_CSLTR_DIR}")

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    total_processed = 0

    for folder_name in folders:
        class_name = folder_name.strip().lower().replace(' ', '_')
        target_gesture_dir = GESTURE_DATA_DIR / class_name
        target_gesture_dir.mkdir(parents=True, exist_ok=True)
        
        existing_npy = [f for f in os.listdir(target_gesture_dir) if f.endswith(".npy")]
        start_idx = len(existing_npy)
        
        folder_path = ISL_CSLTR_DIR / folder_name
        videos = [v for v in os.listdir(folder_path) if v.lower().endswith(('.mp4', '.mov', '.avi'))]
        
        print(f"\\nProcessing phrase '{class_name}' ({len(videos)} videos)...")
        vids_processed = 0

        for vname in videos:
            vid_path = folder_path / vname
            target_npy_path = target_gesture_dir / f"{start_idx + vids_processed}.npy"

            cap = cv2.VideoCapture(str(vid_path))
            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tot_frames < 10:
                cap.release()
                continue

            target_indices = set(np.linspace(0, tot_frames - 1, 30, dtype=int))
            frames_landmarks = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in target_indices:
                    frame = cv2.flip(frame, 1) # Mirroring logic
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = mp_hands.process(rgb_frame)
                    frames_landmarks.append(extract_landmarks_from_frame(result))
                frame_idx += 1
            cap.release()

            # Pad or truncate to exactly 30 frames
            while len(frames_landmarks) < 30:
                frames_landmarks.append(frames_landmarks[-1] if frames_landmarks else [0.0] * 126)
            frames_landmarks = frames_landmarks[:30]

            np_arr = np.array(frames_landmarks, dtype=np.float32)
            np.save(str(target_npy_path), np_arr)
            vids_processed += 1
            total_processed += 1

        print(f"  Processed {vids_processed} videos for '{class_name}'.")

    mp_hands.close()
    print(f"\\n=======================================================")
    print(f" ALL EXTRACTIONS COMPLETE: {total_processed} videos converted to .npy")
    print(f"=======================================================")

if __name__ == "__main__":
    process_isl_csltr()
