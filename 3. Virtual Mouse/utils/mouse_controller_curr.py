# virtual_mouse.py

from pynput.mouse import Controller
from collections import deque
import math

class VirtualMouseController:
    """
    Controls the system mouse based on gesture data.

    This class translates the relative movement of a detected gesture into
    smooth, scaled mouse cursor movement. It uses ratio-based dampening to
    smoothly decrease sensitivity as the fingers separate, preventing drift on release.
    """

    def __init__(self,
                 scale_factor: float = 2.5,
                 smoothing_buffer_size: int = 5,
                 touch_threshold: float = 1.5,
                 dampening_zone_start: float = 1.2):
        """
        Initializes the VirtualMouseController.

        Args:
            scale_factor: Multiplier to control mouse sensitivity. Higher is faster.
            smoothing_buffer_size: Number of recent movements to average for smoothing.
            touch_threshold: The ratio at which the gesture is considered "released".
            dampening_zone_start: The ratio at which to start dampening movement.
        """
        self.mouse = Controller()
        self.scale_factor = scale_factor
        self.touch_threshold = touch_threshold
        self.dampening_zone_start = dampening_zone_start

        # State variables
        self.previous_midpoint = None
        self.is_gesture_active_prev_frame = False

        # Smoothing buffer
        self.smoothing_buffer = deque(maxlen=smoothing_buffer_size)

    def _calculate_midpoint(self, p1, p2):
        """Calculates the midpoint between two screen coordinate points."""
        if p1 is None or p2 is None:
            return None
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    def move_mouse(self, current_gesture_status: bool, gesture_coords: tuple, ratio: float | None):
        """
        Processes gesture data to move the mouse.

        Args:
            current_gesture_status: A boolean indicating if the gesture is currently active.
            gesture_coords: A tuple containing the (x, y) screen coordinates of the two landmarks.
            ratio: The current smoothed ratio from the gesture detector.
        """
        p1, p2 = gesture_coords
        current_midpoint = self._calculate_midpoint(p1, p2)

        if current_gesture_status and current_midpoint:
            if not self.is_gesture_active_prev_frame:
                self.previous_midpoint = current_midpoint
                self.smoothing_buffer.clear()
            else:
                if self.previous_midpoint:
                    dx = current_midpoint[0] - self.previous_midpoint[0]
                    dy = current_midpoint[1] - self.previous_midpoint[1]
                    self.smoothing_buffer.append((dx, dy))

                    if self.smoothing_buffer:
                        avg_dx = sum(item[0] for item in self.smoothing_buffer) / len(self.smoothing_buffer)
                        avg_dy = sum(item[1] for item in self.smoothing_buffer) / len(self.smoothing_buffer)

                        # --- NEW: RATIO-BASED DAMPENING LOGIC ---
                        dampening_factor = 1.0
                        if ratio is not None and ratio > self.dampening_zone_start:
                            # Calculate how far the ratio is into the dampening zone
                            zone_width = self.touch_threshold - self.dampening_zone_start
                            progress_in_zone = ratio - self.dampening_zone_start
                            
                            # Calculate dampening factor (scales from 1.0 down to 0.0)
                            dampening_factor = 1 - (progress_in_zone / zone_width)
                            # Clamp the value between 0 and 1 to be safe
                            dampening_factor = max(0.0, min(1.0, dampening_factor))

                        # Apply the dampening factor to the movement
                        final_dx = avg_dx * dampening_factor
                        final_dy = avg_dy * dampening_factor
                        # --- END OF NEW LOGIC ---

                        self.mouse.move(final_dx * self.scale_factor, final_dy * self.scale_factor)

                self.previous_midpoint = current_midpoint

        self.is_gesture_active_prev_frame = current_gesture_status
