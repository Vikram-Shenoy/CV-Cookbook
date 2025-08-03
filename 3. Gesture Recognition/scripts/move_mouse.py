import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

def main():
    """
    Main function to run the improved hand tracking mouse control with relative movement.
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

    # --- Control Variables ---
    # Sensitivity factor for mouse movement. Higher value = more sensitive.
    sensitivity = 1
    
    # Activation delay variables
    gesture_start_time = None
    activation_delay = 1.0  # 1-second delay
    is_mouse_active = False
    
    # Variables for relative movement
    is_first_active_frame = True
    prev_finger_x, prev_finger_y = 0, 0

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
            is_index_up = landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_MCP].y
            is_middle_down = landmarks[MIDDLE_FINGER_TIP].y > landmarks[MIDDLE_FINGER_PIP].y
            is_ring_down = landmarks[RING_FINGER_TIP].y > landmarks[RING_FINGER_PIP].y
            is_pinky_down = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y

            is_pointing_gesture = is_index_up and is_middle_down and is_ring_down and is_pinky_down

            # --- Handle Activation Delay & Relative Movement ---
            if is_pointing_gesture:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                
                elapsed_time = time.time() - gesture_start_time
                index_tip = landmarks[INDEX_FINGER_TIP]
                index_tip_coords_on_frame = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))

                if elapsed_time >= activation_delay:
                    is_mouse_active = True
                    current_finger_x = index_tip.x
                    current_finger_y = index_tip.y
                    
                    # --- Relative Mouse Movement Logic ---
                    if is_first_active_frame:
                        # On the first frame, set the anchor point and don't move the mouse
                        prev_finger_x = current_finger_x
                        prev_finger_y = current_finger_y
                        is_first_active_frame = False
                    else:
                        # Calculate the change in finger position (delta)
                        delta_x = current_finger_x - prev_finger_x
                        delta_y = current_finger_y - prev_finger_y

                        # Scale the delta by screen size and sensitivity
                        mouse_move_x = delta_x * screen_width * sensitivity
                        mouse_move_y = delta_y * screen_height * sensitivity
                        
                        # Move the mouse relatively
                        pyautogui.move(mouse_move_x, mouse_move_y)

                        # Update the previous finger position for the next frame's calculation
                        prev_finger_x = current_finger_x
                        prev_finger_y = current_finger_y
                    
                    # Visual feedback for active state (green circle)
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 0), cv2.FILLED)
                else:
                    # During the delay, reset the active state and show yellow circle
                    is_mouse_active = False
                    is_first_active_frame = True # Reset for the next activation
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 255), cv2.FILLED)
            else:
                # If the gesture is lost, reset everything
                gesture_start_time = None
                is_mouse_active = False
                is_first_active_frame = True

            # Draw the hand landmarks on the frame for debugging and visualization
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Display the output ---
        # cv2.imshow('Hand Tracking Mouse Control', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == '__main__':
    main()
