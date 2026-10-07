import numpy as np
from ml.recognition.recognizer import SignRecognizer

rec = SignRecognizer()
data = np.load('data/gesture_data/home/0.npy')

for i in range(40):
    idx = i if i < 30 else 29
    features = data[idx].tolist()
    has_hand = any(f != 0.0 for f in features)
    res = rec.process_landmarks_data(features, has_hand)
    if res['current_gesture'] != 'none':
        print(f"Frame {i}: current={res['current_gesture']} conf={res['confidence']} event={res['event']}")
