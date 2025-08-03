# main.py
# To install necessary libraries, run the following commands in your terminal:
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import math
from collections import deque

# --- Landmark Indices ---
LANDMARK_INDICES = {
    "thumb_tip": 4,
    "index_tip": 8,
    "index_knuckle": 5,
    "middle_knuckle": 9,
}

# --- Configuration ---
# The ratio threshold for determining a "touch".
TOUCHING_RATIO_THRESHOLD = 1.5
# The number of frames to average over for smoothing. A larger number means more smoothing but more delay.
HISTORY_BUFFER_SIZE = 10


def draw_text_with_background(image, text, position, font, scale, color, bg_color):
    """Draws text with a solid background color for better visibility."""
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 2)
    bg_rect_start = (position[0] - 10, position[1] + 10)
    bg_rect_end = (position[0] + text_width + 10, position[1] - text_height - 10)
    cv2.rectangle(image, bg_rect_start, bg_rect_end, bg_color, -1)
    cv2.putText(image, text, position, font, scale, color, 2)

def calculate_3d_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points (landmarks)."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    """
    Main function to capture video, detect hand landmarks, and use a smoothed,
    normalized ratio to determine if the thumb and index finger are touching.
    """
    # Initialize MediaPipe Hands solution.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- Initialize History Buffer ---
    # A deque is used for efficiently storing the last N measurements.
    ratio_history = deque(maxlen=HISTORY_BUFFER_SIZE)

    # Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("--- Dynamic Thumb-Index Distance Tracker Running ---")
    print("Using temporal smoothing for stable touch detection.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # Extract landmarks for calculation
            thumb_tip = hand_landmarks.landmark[LANDMARK_INDICES["thumb_tip"]]
            index_tip = hand_landmarks.landmark[LANDMARK_INDICES["index_tip"]]
            index_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["index_knuckle"]]
            middle_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["middle_knuckle"]]

            # Calculate 3D distances
            target_dist = calculate_3d_distance(thumb_tip, index_tip)
            reference_dist = calculate_3d_distance(index_knuckle, middle_knuckle)

            if reference_dist > 0:
                # Calculate the instantaneous ratio
                ratio = target_dist / reference_dist
                
                # --- Temporal Smoothing ---
                # Add the current ratio to our history buffer.
                ratio_history.append(ratio)

                # Only make a decision if the buffer is full to ensure stability.
                if len(ratio_history) == HISTORY_BUFFER_SIZE:
                    # Calculate the smoothed ratio by averaging the history.
                    smoothed_ratio = sum(ratio_history) / len(ratio_history)

                    # --- Determine Touching Status based on Smoothed Ratio ---
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bg_color = (0, 0, 0)

                    # Display the smoothed ratio for tuning.
                    ratio_text = f"Smoothed Ratio: {smoothed_ratio:.2f}"
                    draw_text_with_background(image, ratio_text, (50, 50), font, 1, (255, 255, 255), bg_color)

                    if smoothed_ratio < TOUCHING_RATIO_THRESHOLD:
                        status_text = "Status: Touching"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = "Status: Away"
                        status_color = (255, 255, 255)  # White
                    
                    draw_text_with_background(image, status_text, (50, 100), font, 1, status_color, bg_color)

            # Draw line between fingertips for visualization
            thumb_pixel_x = math.floor(thumb_tip.x * image_width)
            thumb_pixel_y = math.floor(thumb_tip.y * image_height)
            index_pixel_x = math.floor(index_tip.x * image_width)
            index_pixel_y = math.floor(index_tip.y * image_height)
            cv2.line(image, (thumb_pixel_x, thumb_pixel_y), (index_pixel_x, index_pixel_y), (0, 255, 0), 3)

        cv2.imshow('Dynamic Thumb-Index Tracker', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Cleanup
    print("\nExiting program.")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
