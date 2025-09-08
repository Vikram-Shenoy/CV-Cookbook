# main.py

import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gesture_detector import GestureDetector
from utils.mouse_controller import VirtualMouseController
from pynput.mouse import Button # We might need this if we extend functionality

# --- Configuration ---
# Gesture tracking configuration
TARGET_LANDMARK_1 = 4
TARGET_LANDMARK_2 = 8
REF_LANDMARK_1 = 0
REF_LANDMARK_2 = 17
TOUCHING_RATIO_THRESHOLD = 0.33
HISTORY_BUFFER_SIZE = 10

# Application logic configuration
CLICK_FRAME_THRESHOLD = 30 # How many frames a pinch must last to be a "move" instead of a "click"

# Mouse controller configuration
MOUSE_SCALE_FACTOR = 2.5 # Adjust this for mouse sensitivity
MOUSE_SMOOTHING_BUFFER_SIZE = 5

def main():
    """
    Main function to run the virtual mouse application with click and drag.
    """
    # Initialize the detector and controller
    detector = GestureDetector(
        target_landmark1=TARGET_LANDMARK_1,
        target_landmark2=TARGET_LANDMARK_2,
        ref_landmark1=REF_LANDMARK_1,
        ref_landmark2=REF_LANDMARK_2,
        touch_threshold=TOUCHING_RATIO_THRESHOLD,
        buffer_size=HISTORY_BUFFER_SIZE
    )
    mouse_controller = VirtualMouseController(
        scale_factor=MOUSE_SCALE_FACTOR,
        smoothing_buffer_size=MOUSE_SMOOTHING_BUFFER_SIZE,
        touch_threshold=TOUCHING_RATIO_THRESHOLD,
        dampening_zone_start=0.15
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- State Machine Variables ---
    current_state = "IDLE" # Can be "IDLE", "PINCH_DETECTED", or "MOVING"
    frame_counter = 0
    pinch_start_frame = 0

    print("--- Virtual Mouse Running ---")
    print("Quick pinch and release to CLICK.")
    print("Pinch and hold to MOVE.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_counter += 1
        image = cv2.flip(image, 1)

        # Get gesture results from the detector
        # Process frame
        result = detector.process_frame(image, draw=True)
        is_pinching = result["status"]

        # --- State Machine Logic ---

        if current_state == "IDLE":
            if is_pinching:
                # Transition: Pinch detected
                current_state = "PINCH_DETECTED"
                pinch_start_frame = frame_counter
                print("State change: IDLE -> PINCH_DETECTED")

        elif current_state == "PINCH_DETECTED":
            pinch_duration = frame_counter - pinch_start_frame

            if not is_pinching:
                # Short pinch = Click
                print("Action: CLICK")
                mouse_controller.click()
                current_state = "IDLE"
                print("State change: PINCH_DETECTED -> IDLE")

            elif pinch_duration > CLICK_FRAME_THRESHOLD:
                # Long pinch = Move
                current_state = "MOVING"
                print("State change: PINCH_DETECTED -> MOVING")

        elif current_state == "MOVING":
            if is_pinching:
                # Continue moving mouse
                mouse_controller.move_mouse(
                    current_gesture_status=True,
                    gesture_coords=result.get("target_coords"),
                    ratio=result.get("ratio"),
                )
            else:
                # Pinch released -> back to idle
                current_state = "IDLE"
                print("State change: MOVING -> IDLE")


        cv2.imshow('Virtual Mouse', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
