import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

def main():
    """
    Main function to run the improved hand tracking mouse control.
    """
    # --- Setup ---
    # Disable PyAutoGUI's fail-safe to allow moving the mouse to the screen corners
    pyautogui.FAILSAFE = False

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Configure Hands for one hand, with specific confidence levels for detection and tracking
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()

    # Initialize OpenCV Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Smoothing and Delay Variables ---
    # Increased smoothing factor to reduce cursor jitter
    smoothing = 10
    # Previous mouse coordinates
    prev_x, prev_y = 0, 0
    
    # Activation delay variables
    gesture_start_time = None
    activation_delay = 1.0  # 1-second delay
    is_mouse_active = False

    print("Starting Hand Tracking Mouse Control. Press 'q' to quit.")
    print("Hold index finger out for 1 second to activate mouse control.")

    # --- Main Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a more intuitive selfie-view display
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(rgb_frame)

        # --- Hand Landmark Processing ---
        if results.multi_hand_landmarks:
            # Get landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            # --- More Robust Gesture Recognition ---
            # Landmark IDs for fingertips and key joints
            INDEX_FINGER_TIP = 8
            INDEX_FINGER_MCP = 5  # Knuckle at the base of the index finger
            MIDDLE_FINGER_TIP = 12
            MIDDLE_FINGER_PIP = 10
            RING_FINGER_TIP = 16
            RING_FINGER_PIP = 14
            PINKY_TIP = 20
            PINKY_PIP = 18

            # Check if index finger is extended and others are closed
            # This is more robust to hand rotation
            is_index_up = landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_MCP].y
            is_middle_down = landmarks[MIDDLE_FINGER_TIP].y > landmarks[MIDDLE_FINGER_PIP].y
            is_ring_down = landmarks[RING_FINGER_TIP].y > landmarks[RING_FINGER_PIP].y
            is_pinky_down = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y

            is_pointing_gesture = is_index_up and is_middle_down and is_ring_down and is_pinky_down

            # --- Handle Activation Delay ---
            if is_pointing_gesture:
                # If gesture has just started, record the time
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                
                elapsed_time = time.time() - gesture_start_time

                # Get the coordinates of the index finger tip
                index_tip = landmarks[INDEX_FINGER_TIP]
                index_tip_coords_on_frame = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))

                if elapsed_time >= activation_delay:
                    is_mouse_active = True
                    # --- Mouse Movement ---
                    # Convert normalized coordinates to screen coordinates
                    target_x = np.interp(index_tip.x, (0.1, 0.9), (0, screen_width))
                    target_y = np.interp(index_tip.y, (0.1, 0.9), (0, screen_height))

                    # Smooth the mouse movement
                    current_x = prev_x + (target_x - prev_x) / smoothing
                    current_y = prev_y + (target_y - prev_y) / smoothing
                    
                    # Move the mouse
                    pyautogui.moveTo(current_x, current_y)

                    # Update previous coordinates for the next frame
                    prev_x, prev_y = current_x, current_y
                    
                    # Visual feedback for active state (green circle)
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 0), cv2.FILLED)
                else:
                    # Visual feedback for delay period (yellow circle)
                    is_mouse_active = False
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 255), cv2.FILLED)
            else:
                # Gesture is not detected, reset timer and state
                gesture_start_time = None
                is_mouse_active = False

            # Draw the hand landmarks on the frame for debugging and visualization
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Display the output ---
        cv2.imshow('Hand Tracking Mouse Control', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == '__main__':
    main()
