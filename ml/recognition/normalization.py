"""
Standardized Landmark Normalization for Signova Recognition Pipeline
Provides invariant mathematical representation of MediaPipe hand landmarks:
- Translation-invariant (wrist-centered)
- Scale/distance-invariant (normalized by palm span ||Wrist - Middle_MCP||)
- Relative inter-hand orientation & distance encoding for two-handed gestures
- Shared identically across training, validation, and real-time inference
"""
from __future__ import annotations

import numpy as np
from typing import List, Union

def normalize_landmarks_frame(frame_126: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Normalizes a single frame of 126 floats (63 Left Hand + 63 Right Hand).
    Input: shape (126,)
    Output: shape (126,) float32 normalized features
    """
    if isinstance(frame_126, list):
        frame_126 = np.array(frame_126, dtype=np.float32)
    elif frame_126.dtype != np.float32:
        frame_126 = frame_126.astype(np.float32)

    norm = np.zeros(126, dtype=np.float32)
    lh = frame_126[:63]
    rh = frame_126[63:]

    lh_present = bool(np.any(lh != 0.0))
    rh_present = bool(np.any(rh != 0.0))

    # Phase 2: Canonical Landmark Representation
    # Map active hand to primary slot [0:63]
    primary_raw = np.zeros(63, dtype=np.float32)
    secondary_raw = np.zeros(63, dtype=np.float32)
    
    # Mirror X coordinates for LH if it's the primary hand, so its geometry matches a RH
    if rh_present and not lh_present:
        primary_raw = rh
    elif lh_present and not rh_present:
        primary_raw = lh.copy()
        primary_raw[0::3] = -primary_raw[0::3] # mirror X
    elif lh_present and rh_present:
        # Two handed gesture: preserve both, deterministic ordering (RH primary, LH secondary mirrored)
        primary_raw = rh
        secondary_raw = lh.copy()
        secondary_raw[0::3] = -secondary_raw[0::3]
        
    primary_present = bool(np.any(primary_raw != 0.0))
    secondary_present = bool(np.any(secondary_raw != 0.0))
    
    p_scale = 1.0
    s_scale = 1.0

    if primary_present:
        px0, py0, pz0 = primary_raw[0], primary_raw[1], primary_raw[2]
        px9, py9, pz9 = primary_raw[27], primary_raw[28], primary_raw[29]
        p_scale = float(np.sqrt((px9 - px0)**2 + (py9 - py0)**2 + (pz9 - pz0)**2))
        if p_scale < 1e-4:
            p_scale = 1.0

        for k in range(21):
            norm[3 * k] = (primary_raw[3 * k] - px0) / p_scale
            norm[3 * k + 1] = (primary_raw[3 * k + 1] - py0) / p_scale
            norm[3 * k + 2] = (primary_raw[3 * k + 2] - pz0) / p_scale

    if secondary_present:
        sx0, sy0, sz0 = secondary_raw[0], secondary_raw[1], secondary_raw[2]
        sx9, sy9, sz9 = secondary_raw[27], secondary_raw[28], secondary_raw[29]
        s_scale = float(np.sqrt((sx9 - sx0)**2 + (sy9 - sy0)**2 + (sz9 - sz0)**2))
        if s_scale < 1e-4:
            s_scale = 1.0

        for k in range(21):
            norm[63 + 3 * k] = (secondary_raw[3 * k] - sx0) / s_scale
            norm[63 + 3 * k + 1] = (secondary_raw[3 * k + 1] - sy0) / s_scale
            norm[63 + 3 * k + 2] = (secondary_raw[3 * k + 2] - sz0) / s_scale

    # Encoding relative orientation for two handed
    if primary_present and secondary_present:
        avg_scale = (p_scale + s_scale) / 2.0
        norm[63] = (secondary_raw[0] - primary_raw[0]) / avg_scale
        norm[64] = (secondary_raw[1] - primary_raw[1]) / avg_scale
        norm[65] = (secondary_raw[2] - primary_raw[2]) / avg_scale

    return norm


def normalize_landmarks_sequence(seq_30x126: np.ndarray) -> np.ndarray:
    """
    Normalizes a temporal sequence of shape (T, 126) or (N, T, 126).
    """
    if seq_30x126.ndim == 2:
        out = np.zeros_like(seq_30x126, dtype=np.float32)
        for t in range(seq_30x126.shape[0]):
            out[t] = normalize_landmarks_frame(seq_30x126[t])
        return out
    elif seq_30x126.ndim == 3:
        out = np.zeros_like(seq_30x126, dtype=np.float32)
        for i in range(seq_30x126.shape[0]):
            for t in range(seq_30x126.shape[1]):
                out[i, t] = normalize_landmarks_frame(seq_30x126[i, t])
        return out
    else:
        raise ValueError(f"Unexpected array shape for landmark normalization: {seq_30x126.shape}")
