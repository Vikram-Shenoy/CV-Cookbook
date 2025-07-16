# main.py
# To install necessary libraries, run the following commands in your terminal:
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import math

# --- Landmark Indices for Fingertips ---
# This dictionary maps finger names to the MediaPipe landmark index for the tip of that finger.
# Refer to the MediaPipe hand landmarks model for a visual representation:
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
FINGERTIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# --- Touching Threshold ---
# If the distance in pixels is less than this value, the fingers are considered "touching".
# You may need to adjust this value based on your camera and hand size.
TOUCHING_THRESHOLD = 75

def draw_text_with_background(image, text, position, font, scale, color, bg_color):
    """
    Draws text with a solid background color for better visibility.
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 2)
    bg_rect_start = (position[0] - 10, position[1] + 10)
    bg_rect_end = (position[0] + text_width + 10, position[1] - text_height - 10)
    cv2.rectangle(image, bg_rect_start, bg_rect_end, bg_color, -1)
    cv2.putText(image, text, position, font, scale, color, 2)


def main():
    """
    Main function to capture video, detect hand landmarks, and display coordinates
    and distance between the thumb and index finger.
    """
    # Initialize MediaPipe Hands solution.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Track only one hand for simplicity
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- Video Capture ---
    # Use 0 for the default webcam. If you have multiple cameras, you might need to change this.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("--- Thumb-Index Distance Tracker Running ---")
    print("Tracking the distance between the Thumb and Index finger.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        # Read a frame from the webcam.
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB for MediaPipe.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks.
        results = hands.process(image_rgb)

        # Get the height and width of the image.
        image_height, image_width, _ = image.shape

        # --- Landmark Processing and Drawing ---
        if results.multi_hand_landmarks:
            # We are tracking only one hand, so we take the first one.
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw the hand landmarks on the image.
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

            # --- Get Coordinates for Thumb and Index Finger ---
            thumb_tip_landmark = hand_landmarks.landmark[FINGERTIP_LANDMARKS["thumb"]]
            index_tip_landmark = hand_landmarks.landmark[FINGERTIP_LANDMARKS["index"]]

            # Convert normalized coordinates to pixel coordinates.
            thumb_pixel_x = math.floor(thumb_tip_landmark.x * image_width)
            thumb_pixel_y = math.floor(thumb_tip_landmark.y * image_height)

            index_pixel_x = math.floor(index_tip_landmark.x * image_width)
            index_pixel_y = math.floor(index_tip_landmark.y * image_height)

            # --- Calculate the Distance ---
            # Use the Euclidean distance formula. math.hypot is efficient for this.
            distance = math.hypot(index_pixel_x - thumb_pixel_x, index_pixel_y - thumb_pixel_y)

            # --- Draw a line between thumb and index finger ---
            cv2.line(image, (thumb_pixel_x, thumb_pixel_y), (index_pixel_x, index_pixel_y), (0, 255, 0), 3)

            # --- Prepare and Display Text on Image ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)  # White
            bg_color = (0, 0, 0)          # Black

            thumb_text = f"Thumb: ({thumb_pixel_x}, {thumb_pixel_y})"
            index_text = f"Index: ({index_pixel_x}, {index_pixel_y})"
            distance_text = f"Distance: {distance:.2f} pixels"

            # draw_text_with_background(image, thumb_text, (50, 50), font, 0.8, font_color, bg_color)
            # draw_text_with_background(image, index_text, (50, 90), font, 0.8, font_color, bg_color)
            draw_text_with_background(image, distance_text, (50, 130), font, 1, font_color, bg_color)

            # --- Determine Touching Status and Display it ---
            if distance < TOUCHING_THRESHOLD:
                status_text = "Status: Touching"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "Status: Away"
                status_color = (255, 255, 255)  # White

            draw_text_with_background(image, status_text, (50, 170), font, 1, status_color, bg_color)


        # --- Display the resulting frame ---
        cv2.imshow('Thumb-Index Distance Tracker', image)

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("\nExiting program.")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
