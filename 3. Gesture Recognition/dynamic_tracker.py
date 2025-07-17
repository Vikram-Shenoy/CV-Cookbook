# main.py
# To install necessary libraries, run the following commands in your terminal:
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import math

# --- Landmark Indices ---
# We define the landmarks we need for our calculations.
# Refer to the MediaPipe hand landmarks model for a visual representation:
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
LANDMARK_INDICES = {
    "thumb_tip": 4,
    "index_tip": 8,
    "index_knuckle": 5,
    "middle_knuckle": 9,
}

# --- Touching Ratio Threshold ---
# If the ratio of (thumb-index distance) / (knuckle distance) is less than this,
# the fingers are considered "touching". This value is more stable than a pixel threshold.
# You might need to fine-tune this value slightly. A good starting point is around 0.6-0.8.
TOUCHING_RATIO_THRESHOLD = 1.5

def draw_text_with_background(image, text, position, font, scale, color, bg_color):
    """
    Draws text with a solid background color for better visibility.
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 2)
    bg_rect_start = (position[0] - 10, position[1] + 10)
    bg_rect_end = (position[0] + text_width + 10, position[1] - text_height - 10)
    cv2.rectangle(image, bg_rect_start, bg_rect_end, bg_color, -1)
    cv2.putText(image, text, position, font, scale, color, 2)

def calculate_3d_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 3D points (landmarks).
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    """
    Main function to capture video, detect hand landmarks, and use a normalized
    ratio to determine if the thumb and index finger are touching.
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

    # Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("--- Dynamic Thumb-Index Distance Tracker Running ---")
    print("Using a normalized ratio for robust touch detection.")
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

            # Draw the hand landmarks on the image for visualization.
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # --- Extract Landmarks for Calculation ---
            # We get the full 3D landmark data.
            thumb_tip = hand_landmarks.landmark[LANDMARK_INDICES["thumb_tip"]]
            index_tip = hand_landmarks.landmark[LANDMARK_INDICES["index_tip"]]
            index_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["index_knuckle"]]
            middle_knuckle = hand_landmarks.landmark[LANDMARK_INDICES["middle_knuckle"]]

            # --- 3D Distance Calculations ---
            # 1. Calculate the distance between the thumb and index finger tips.
            target_dist = calculate_3d_distance(thumb_tip, index_tip)

            # 2. Calculate the reference distance between the two knuckles. This is our "ruler".
            reference_dist = calculate_3d_distance(index_knuckle, middle_knuckle)

            # --- Normalization ---
            # Avoid division by zero if the hand is not detected properly.
            if reference_dist > 0:
                # 3. Calculate the normalized ratio.
                ratio = target_dist / reference_dist

                # --- Determine Touching Status and Display ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                bg_color = (0, 0, 0) # Black

                # Display the calculated ratio for debugging/tuning.
                ratio_text = f"Ratio: {ratio:.2f}"
                draw_text_with_background(image, ratio_text, (50, 50), font, 1, (255, 255, 255), bg_color)

                if ratio < TOUCHING_RATIO_THRESHOLD:
                    status_text = "Status: Touching"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "Status: Away"
                    status_color = (255, 255, 255)  # White

                draw_text_with_background(image, status_text, (50, 100), font, 1, status_color, bg_color)

            # --- Draw a line between thumb and index finger ---
            # For visualization, we still need pixel coordinates.
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
