import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gesture_detector import GestureDetector
# Add this line with your other imports
from utils.mouse_controller_curr import VirtualMouseController
# --- Configuration for this specific use case ---
# Use MediaPipe landmark indices
# Target: Thumb tip to Index tip
"""
FINGERTIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
"""
TARGET_LANDMARK_1 = 4
TARGET_LANDMARK_2 = 8
# Reference: Index knuckle to Middle knuckle
REF_LANDMARK_1 = 17
REF_LANDMARK_2 = 0

TOUCHING_RATIO_THRESHOLD = 0.23
HISTORY_BUFFER_SIZE = 10


# Mouse controller configuration
MOUSE_SCALE_FACTOR = 2.5 # Adjust this for mouse sensitivity
MOUSE_SMOOTHING_BUFFER_SIZE = 5 

def main():
    """
    Main function to run the gesture detection application.
    """
    # Initialize the detector with our specific gesture configuration
    detector = GestureDetector(
        target_landmark1=TARGET_LANDMARK_1,
        target_landmark2=TARGET_LANDMARK_2,
        ref_landmark1=REF_LANDMARK_1,
        ref_landmark2=REF_LANDMARK_2,
        touch_threshold=TOUCHING_RATIO_THRESHOLD,
        buffer_size=HISTORY_BUFFER_SIZE
    )
    # Initialize the mouse controller
    mouse_controller = VirtualMouseController(
        scale_factor=MOUSE_SCALE_FACTOR, 
        smoothing_buffer_size=MOUSE_SMOOTHING_BUFFER_SIZE,
        touch_threshold=TOUCHING_RATIO_THRESHOLD, 
        dampening_zone_start= 0.15
        )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("--- Gesture Detection Running ---")
    print("This script uses the modular GestureDetector class.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        # Process the frame and get the result dictionary
        # We pass draw=True to let the class handle visualization
        result = detector.process_frame(image, draw=True)
        # print(f"Result: {result['status']}, Ratio: {result['ratio']} ,Target Cords: {result['target_coords']}")
        # --- Use the result from the detector in your application ---
        mouse_controller.move_mouse(
            current_gesture_status=result["status"],
            gesture_coords=result["target_coords"],
            ratio=result["ratio"]
            )
        if result["status"]:
            pass
            # Example action: Print a message to the console when touching
            #print(f"Gesture Detected! Smoothed Ratio: {result['ratio']:.2f}")
            # Here you could trigger other actions (e.g., control a mouse, play a sound)
        
        # cv2.imshow('Gesture Detection Example', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Cleanup
    print("\nExiting program.")
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()