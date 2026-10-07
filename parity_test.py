import cv2
import numpy as np
import mediapipe as mp
from ml.recognition.normalization import normalize_landmarks_frame

def get_landmarks_train(frame_bgr):
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    # Training preprocessing
    frame = cv2.flip(frame_bgr, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb_frame)
    
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
    return normalize_landmarks_frame(left_hand + right_hand)

def get_landmarks_live(frame_bgr):
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    # Live preprocessing (backend server-side process_frame_data)
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb_frame)
    
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
    return normalize_landmarks_frame(left_hand + right_hand)

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Add some fake patterns so mediapipe might detect a hand, but wait, mediapipe won't detect a hand on a black image.
