import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from collections import deque

def main():
    """
    Main function to run the advanced hand tracking mouse control, featuring:
    - Relative movement for a trackpad-like feel.
    - Activation delay to prevent accidental input.
    - Position smoothing to eliminate jitter.
    - Mouse acceleration for both fast and precise control.
    """
    # --- Setup ---
    pyautogui.FAILSAFE = False
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Control Variables ---
    
    # -- Activation Delay --
    gesture_start_time = None
    activation_delay = 1.0
    is_mouse_active = False
    
    # -- Relative Movement & Smoothing --
    is_first_active_frame = True
    prev_smoothed_x, prev_smoothed_y = 0, 0
    
    # -- Smoothing --
    smoothing_factor = 10 # Number of frames to average over. Higher = smoother but more lag.
    x_history = deque(maxlen=smoothing_factor)
    y_history = deque(maxlen=smoothing_factor)

    # -- Mouse Acceleration --
    # Sensitivity now dynamically changes based on hand speed.
    min_sensitivity = 0.5
    max_sensitivity = 3.5
    # Speed thresholds (in normalized coordinates per frame).
    min_speed = 0.001
    max_speed = 0.02

    print("Starting Hand Tracking Mouse Control. Press 'q' to quit.")
    print("Hold index finger out for 1 second to activate mouse control.")

    # --- Main Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            # --- Gesture Recognition ---
            INDEX_FINGER_TIP = 8
            INDEX_FINGER_MCP = 5
            MIDDLE_FINGER_TIP = 12
            MIDDLE_FINGER_PIP = 10
            RING_FINGER_TIP = 16
            RING_FINGER_PIP = 14
            PINKY_TIP = 20
            PINKY_PIP = 18

            is_index_up = landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_MCP].y
            is_middle_down = landmarks[MIDDLE_FINGER_TIP].y > landmarks[MIDDLE_FINGER_PIP].y
            is_ring_down = landmarks[RING_FINGER_TIP].y > landmarks[RING_FINGER_PIP].y
            is_pinky_down = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
            is_pointing_gesture = is_index_up and is_middle_down and is_ring_down and is_pinky_down

            if is_pointing_gesture:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                
                elapsed_time = time.time() - gesture_start_time
                index_tip = landmarks[INDEX_FINGER_TIP]
                index_tip_coords_on_frame = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))

                if elapsed_time >= activation_delay:
                    is_mouse_active = True
                    
                    # --- Smoothing Logic ---
                    # Add current finger position to history.
                    x_history.append(index_tip.x)
                    y_history.append(index_tip.y)
                    
                    # Calculate the smoothed position by averaging the history.
                    smoothed_x = np.mean(x_history)
                    smoothed_y = np.mean(y_history)

                    # --- Relative Mouse Movement Logic (using smoothed values) ---
                    if is_first_active_frame:
                        # Set the initial anchor point using the first smoothed position.
                        prev_smoothed_x = smoothed_x
                        prev_smoothed_y = smoothed_y
                        is_first_active_frame = False
                    else:
                        # Calculate change (delta) from the previous smoothed position.
                        delta_x = smoothed_x - prev_smoothed_x
                        delta_y = smoothed_y - prev_smoothed_y

                        # --- Mouse Acceleration Logic ---
                        # Calculate the speed of hand movement.
                        speed = math.sqrt(delta_x**2 + delta_y**2)
                        
                        # Interpolate sensitivity based on speed.
                        # Fast movements get high sensitivity, slow movements get low sensitivity.
                        dynamic_sensitivity = np.interp(speed, [min_speed, max_speed], [min_sensitivity, max_sensitivity])
                        
                        # Scale the delta by screen size and dynamic sensitivity.
                        mouse_move_x = delta_x * screen_width * dynamic_sensitivity
                        mouse_move_y = delta_y * screen_height * dynamic_sensitivity
                        
                        # Move the mouse relatively.
                        pyautogui.move(mouse_move_x, mouse_move_y)

                        # Update the previous position for the next frame's calculation.
                        prev_smoothed_x = smoothed_x
                        prev_smoothed_y = smoothed_y
                    
                    # Visual feedback: green circle for active state.
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 0), cv2.FILLED)
                else:
                    # During activation delay, reset state.
                    is_mouse_active = False
                    is_first_active_frame = True
                    # Visual feedback: yellow circle for "activating".
                    cv2.circle(frame, index_tip_coords_on_frame, 15, (0, 255, 255), cv2.FILLED)
            else:
                # If gesture is lost, reset everything.
                gesture_start_time = None
                is_mouse_active = False
                is_first_active_frame = True
                # Clear history when hand is not in pointing gesture to avoid stale data.
                x_history.clear()
                y_history.clear()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking Mouse Control', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == '__main__':
    main()
