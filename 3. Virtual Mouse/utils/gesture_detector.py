import cv2
import mediapipe as mp
import math
from collections import deque

class GestureDetector:
    """
    A class to detect a specific gesture (e.g., finger touch) using MediaPipe Hands.

    The class is initialized with the landmarks to track and configuration for detection.
    Its main method, process_frame, analyzes a single image frame and returns the
    detection status and other useful data.
    """

    def __init__(self,
                 target_landmark1: int,
                 target_landmark2: int,
                 ref_landmark1: int,
                 ref_landmark2: int,
                 touch_threshold: float = 1.5,
                 buffer_size: int = 10,
                 max_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        """
        Initializes the GestureDetector.

        Args:
            target_landmark1: The index of the first landmark for the target distance.
            target_landmark2: The index of the second landmark for the target distance.
            ref_landmark1: The index of the first landmark for the reference distance.
            ref_landmark2: The index of the second landmark for the reference distance.
            touch_threshold: The ratio threshold to determine a "touch".
            buffer_size: The number of frames to average for smoothing.
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence value for hand detection.
            min_tracking_confidence: Minimum confidence value for hand tracking.
        """
        self.target_landmark1 = target_landmark1
        self.target_landmark2 = target_landmark2
        self.ref_landmark1 = ref_landmark1
        self.ref_landmark2 = ref_landmark2
        self.touch_threshold = touch_threshold
        self.buffer_size = buffer_size

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize history buffer for smoothing
        self.ratio_history = deque(maxlen=self.buffer_size)

    def _calculate_3d_distance(self, p1, p2) -> float:
        """Calculates the Euclidean distance between two 3D landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _draw_text_with_background(self, image, text, position, font, scale, color, bg_color):
        """Helper to draw text with a solid background."""
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 2)
        bg_rect_start = (position[0] - 10, position[1] + 10)
        bg_rect_end = (position[0] + text_width + 10, position[1] - text_height - 10)
        cv2.rectangle(image, bg_rect_start, bg_rect_end, bg_color, -1)
        cv2.putText(image, text, position, font, scale, color, 2)


    def process_frame(self, image, draw: bool = True) -> dict:
        """
        Processes a single video frame to detect the gesture.

        Args:
            image: The input image frame from OpenCV.
            draw: If True, draws visualization on the image.

        Returns:
            A dictionary containing detection results:
            {
                "status": bool,
                "ratio": float | None,
                "target_coords": tuple | None,
                "hand_landmarks": LandmarkList | None
            }
        """
        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        # Default result dictionary
        result = {
            "status": False,
            "ratio": None,
            "target_coords": (None, None),
            "hand_landmarks": None
        }

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            result["hand_landmarks"] = hand_landmarks

            # Extract landmarks for calculation
            p1 = hand_landmarks.landmark[self.target_landmark1]
            p2 = hand_landmarks.landmark[self.target_landmark2]
            ref1 = hand_landmarks.landmark[self.ref_landmark1]
            ref2 = hand_landmarks.landmark[self.ref_landmark2]

            # Store pixel coordinates of target landmarks
            p1_px = (math.floor(p1.x * image_width), math.floor(p1.y * image_height))
            p2_px = (math.floor(p2.x * image_width), math.floor(p2.y * image_height))
            result["target_coords"] = (p1_px, p2_px)

            # Calculate distances and ratio
            target_dist = self._calculate_3d_distance(p1, p2)
            reference_dist = self._calculate_3d_distance(ref1, ref2)

            if reference_dist > 0:
                ratio = target_dist / reference_dist
                self.ratio_history.append(ratio)

                # Wait for the buffer to be full for stable results
                if len(self.ratio_history) == self.buffer_size:
                    smoothed_ratio = sum(self.ratio_history) / len(self.ratio_history)
                    result["ratio"] = smoothed_ratio
                    result["status"] = smoothed_ratio < self.touch_threshold

        # Optional drawing
        if draw:
            # Draw full hand landmarks if detected
            if result["hand_landmarks"]:
                self.mp_drawing.draw_landmarks(
                    image, result["hand_landmarks"], self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            # Draw line between target landmarks
            if all(result["target_coords"]):
                cv2.line(image, result["target_coords"][0], result["target_coords"][1], (0, 255, 0), 3)

            # Draw status and ratio text
            font = cv2.FONT_HERSHEY_SIMPLEX
            bg_color = (0, 0, 0)
            
            if result["ratio"] is not None:
                ratio_text = f"Smoothed Ratio: {result['ratio']:.2f}"
                self._draw_text_with_background(image, ratio_text, (50, 50), font, 1, (255, 255, 255), bg_color)
            
            status_text = "Status: Touching" if result["status"] else "Status: Away"
            status_color = (0, 255, 0) if result["status"] else (255, 255, 255)
            self._draw_text_with_background(image, status_text, (50, 100), font, 1, status_color, bg_color)


        return result

    def close(self):
        """Releases the MediaPipe Hands resources."""
        self.hands.close()