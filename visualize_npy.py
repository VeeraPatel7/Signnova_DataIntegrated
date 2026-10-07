import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Connections between landmarks for a MediaPipe hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

def visualize_npy(npy_path, output_gif_path):
    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    # data shape: (frames, 126) -> 30 frames, 126 values (63 left hand, 63 right hand)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # We want to flip Y because mediapipe Y is from top to bottom
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("Hand Gesture Visualization")
    
    lines_left = [ax.plot([], [], 'ro-', lw=2, markersize=3)[0] for _ in range(len(HAND_CONNECTIONS))]
    lines_right = [ax.plot([], [], 'bo-', lw=2, markersize=3)[0] for _ in range(len(HAND_CONNECTIONS))]

    def init():
        for line in lines_left + lines_right:
            line.set_data([], [])
        return lines_left + lines_right

    def update(frame_idx):
        frame_data = data[frame_idx]
        
        # Left hand (first 63 values -> 21 points x 3 coords)
        left_hand = frame_data[:63].reshape((21, 3))
        # Right hand (next 63 values -> 21 points x 3 coords)
        right_hand = frame_data[63:].reshape((21, 3))
        
        for i, (start, end) in enumerate(HAND_CONNECTIONS):
            # Left hand (Red)
            if left_hand[start][0] != 0 or left_hand[start][1] != 0:
                lines_left[i].set_data(
                    [left_hand[start][0], left_hand[end][0]],
                    [left_hand[start][1], left_hand[end][1]]
                )
            else:
                lines_left[i].set_data([], [])
                
            # Right hand (Blue)
            if right_hand[start][0] != 0 or right_hand[start][1] != 0:
                lines_right[i].set_data(
                    [right_hand[start][0], right_hand[end][0]],
                    [right_hand[start][1], right_hand[end][1]]
                )
            else:
                lines_right[i].set_data([], [])

        return lines_left + lines_right

    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True, interval=100)
    
    print(f"Saving to {output_gif_path}...")
    ani.save(output_gif_path, writer='pillow', fps=10)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    visualize_npy("data/gesture_data/accident/0.npy", "accident_sign_visualization.gif")
