# main.py
# To install necessary libraries, run the following commands in your terminal:
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import math

# --- Finger Selection ---
# You can change this value to track a different finger.
# Options: "thumb", "index", "middle", "ring", "pinky"
SELECTED_FINGER = "thumb"

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

def main():
    """
    Main function to capture video, detect hand landmarks, and display coordinates.
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

    print("--- Finger Tracker Running ---")
    print(f"Tracking: {SELECTED_FINGER.capitalize()} Finger (Tip Landmark: {FINGERTIP_LANDMARKS[SELECTED_FINGER]})")
    print("Press 'q' to quit.")

    while cap.isOpened():
        # Read a frame from the webcam.
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display.
        # This makes the video feed feel more natural like a mirror.
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB.
        # MediaPipe requires RGB input.
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

            # --- Get and Display Coordinates ---
            try:
                # Get the landmark index for the selected finger.
                fingertip_index = FINGERTIP_LANDMARKS[SELECTED_FINGER.lower()]

                # Get the normalized coordinates of the fingertip.
                fingertip_landmark = hand_landmarks.landmark[fingertip_index]

                # Convert normalized coordinates to pixel coordinates.
                # The x and y values from MediaPipe are normalized (0.0 to 1.0).
                # We multiply by the image width and height to get the actual pixel values.
                pixel_x = math.floor(fingertip_landmark.x * image_width)
                pixel_y = math.floor(fingertip_landmark.y * image_height)

                # Prepare the text to display.
                coord_text = f"{SELECTED_FINGER.capitalize()}: ({pixel_x}, {pixel_y})"
                print(f"\r{coord_text}", end="") # Print to console without newlines

                # --- Display Text on Image ---
                # Position the text on the screen.
                text_position = (50, 50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 255) # White
                line_type = 2
                text_background_color = (0, 0, 0) # Black

                # Add a black background rectangle for better text visibility.
                (text_width, text_height), _ = cv2.getTextSize(coord_text, font, font_scale, line_type)
                cv2.rectangle(image, (text_position[0] - 10, text_position[1] + 10),
                              (text_position[0] + text_width + 10, text_position[1] - text_height - 10),
                              text_background_color, -1)


                # Put the coordinate text on the image.
                cv2.putText(image, coord_text, text_position, font, font_scale, font_color, line_type)

            except KeyError:
                # Handle cases where an invalid finger name is provided.
                error_text = f"Error: Invalid finger '{SELECTED_FINGER}'. Check options."
                cv2.putText(image, error_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # --- Display the resulting frame ---
        cv2.imshow('MediaPipe Hand Tracking', image)

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
