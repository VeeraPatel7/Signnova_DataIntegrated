import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
GESTURE_DATA_DIR = REPO_ROOT / "data" / "gesture_data"
AGRIDATA_DIR = REPO_ROOT / "data" / "agridata"

def extract_landmarks_from_frame(result):
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                right_hand = landmarks
            else:
                left_hand = landmarks

    return left_hand + right_hand

def process_agridata():
    if not AGRIDATA_DIR.exists():
        raise FileNotFoundError(f"Directory not found at {AGRIDATA_DIR}")

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    total_processed = 0
    class_counts = {}

    # Gather all avi files
    all_videos = []
    for root, dirs, files in os.walk(AGRIDATA_DIR):
        for f in files:
            if f.lower().endswith('.avi'):
                all_videos.append(os.path.join(root, f))
                
    print(f"Found {len(all_videos)} total videos in {AGRIDATA_DIR}")

    for vid_path in all_videos:
        filename = os.path.basename(vid_path)
        class_name = filename.split('_')[0].lower().strip()
        
        # fix typo in dataset
        if class_name == 'cococnut':
            class_name = 'coconut'
            
        target_gesture_dir = GESTURE_DATA_DIR / class_name
        target_gesture_dir.mkdir(parents=True, exist_ok=True)
        
        existing_npy = [f for f in os.listdir(target_gesture_dir) if f.endswith(".npy")]
        start_idx = len(existing_npy)
        target_npy_path = target_gesture_dir / f"{start_idx}.npy"

        cap = cv2.VideoCapture(vid_path)
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot_frames < 5:
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

        while len(frames_landmarks) < 30:
            frames_landmarks.append(frames_landmarks[-1] if frames_landmarks else [0.0] * 126)
        frames_landmarks = frames_landmarks[:30]

        np_arr = np.array(frames_landmarks, dtype=np.float32)
        np.save(str(target_npy_path), np_arr)
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_processed += 1
        
        if total_processed % 50 == 0:
            print(f"Processed {total_processed}/{len(all_videos)} videos...")

    mp_hands.close()
    print(f"\\n=======================================================")
    print(f" ALL EXTRACTIONS COMPLETE: {total_processed} videos converted to .npy")
    print(" Class breakdown:", class_counts)
    print(f"=======================================================")

if __name__ == "__main__":
    process_agridata()
