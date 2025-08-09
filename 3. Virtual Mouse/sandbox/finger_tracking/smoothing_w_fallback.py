"""
After testing the following fallback logic:
# Attempt 1 (Primary - Palm):

First, check the visibility score for both the index knuckle (5) and the pinky knuckle (17).

Only if both landmarks have a high visibility score (e.g., > 0.5), do we trust the "Palm" reference.

# Attempt 2 (Fallback - Knuckle):

If the visibility check for the palm fails (meaning the pinky is likely hidden), we then proceed to the fallback.

We check the visibility score for the index knuckle (5) and the middle finger knuckle (9).

If these are visible, we use the "Knuckle" reference.

# Failure Case:

If neither set of landmarks is clearly visible, we don't have a reliable ruler for that frame, and we hold the last known status.

## Conclusion:
This approach doesn't work well in practice, as mediapipe guesses the visibility of landmarks based on their 
position in the image, which can lead to false positives or negatives. 
Instead, we will simplify the logic to use a single palm reference and apply temporal smoothing to 
determine touch status.

"""

import cv2
import mediapipe as mp
import math
from collections import deque

# --- Landmark Indices ---
LANDMARK_INDICES = {
    "thumb_tip": 4,
    "index_tip": 8,
    "index_knuckle": 5,
    "pinky_knuckle": 17,
}

# --- Configuration ---
# The ratio threshold for determining a "touch" using the palm width as a reference.
TOUCHING_RATIO_THRESHOLD = 0.7
# The number of frames to average over for smoothing.
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
    Main function using a single palm reference and temporal smoothing
    for robust touch detection.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- State Variables ---
    ratio_history = deque(maxlen=HISTORY_BUFFER_SIZE)
    current_status = "Away" # Hold the last known status

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("--- Simplified Thumb-Index Tracker Running ---")
    print("Using Palm reference and smoothing.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image_height, image_width, _ = image.shape

        font = cv2.FONT_HERSHEY_SIMPLEX # Define font here for use in all text drawing

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # --- Extract all necessary landmarks ---
            thumb_tip = hand_landmarks.landmark[LANDMARK_INDICES["thumb_tip"]]
            index_tip = hand_landmarks.landmark[LANDMARK_INDICES["index_tip"]]
            index_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["index_knuckle"]]
            pinky_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["pinky_knuckle"]]

            # --- Calculate Reference and Ratio ---
            # Use the palm width (index to pinky knuckle) as the single reference.
            reference_dist = calculate_3d_distance(index_knuckle, pinky_knuckle)

            if reference_dist > 0.05: # Basic check for valid detection
                target_dist = calculate_3d_distance(thumb_tip, index_tip)
                ratio = target_dist / reference_dist
                ratio_history.append(ratio)

                if len(ratio_history) == HISTORY_BUFFER_SIZE:
                    smoothed_ratio = sum(ratio_history) / len(ratio_history)
                    if smoothed_ratio < TOUCHING_RATIO_THRESHOLD:
                        current_status = "Touching"
                    else:
                        current_status = "Away"

                    # Display the smoothed ratio
                    ratio_text = f"Smoothed Ratio: {smoothed_ratio:.2f}"
                    draw_text_with_background(image, ratio_text, (50, 50), font, 1, (255, 255, 255), (0,0,0))
            
            # Draw the status text regardless of whether a new measurement was made
            status_color = (0, 255, 0) if current_status == "Touching" else (255, 255, 255)
            draw_text_with_background(image, f"Status: {current_status}", (50, 100), font, 1, status_color, (0, 0, 0))

            # Draw line for visualization
            thumb_pixel_x = math.floor(thumb_tip.x * image_width)
            thumb_pixel_y = math.floor(thumb_tip.y * image_height)
            index_pixel_x = math.floor(index_tip.x * image_width)
            index_pixel_y = math.floor(index_tip.y * image_height)
            cv2.line(image, (thumb_pixel_x, thumb_pixel_y), (index_pixel_x, index_pixel_y), (0, 255, 0), 3)

        cv2.imshow('Simplified Thumb-Index Tracker', image)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    # Cleanup
    print("\nExiting program.")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
