# virtual_mouse.py

from pynput.mouse import Controller
from collections import deque
import math

class VirtualMouseController:
    """
    Controls the system mouse based on gesture data.

    This class translates the relative movement of a detected gesture into
    smooth, scaled mouse cursor movement, ignoring minor jitters within a
    defined deadzone.
    """

    def __init__(self,
                 scale_factor: float = 2.5,
                 smoothing_buffer_size: int = 5,
                 deadzone_radius: float = 1.5):
        """
        Initializes the VirtualMouseController.

        Args:
            scale_factor: Multiplier to control mouse sensitivity. Higher is faster.
            smoothing_buffer_size: Number of recent movements to average for smoothing.
            deadzone_radius: The radius (in pixels) within which movement is ignored.
        """
        self.mouse = Controller()
        self.scale_factor = scale_factor
        self.deadzone_radius = deadzone_radius

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

    def move_mouse(self, current_gesture_status: bool, gesture_coords: tuple):
        """
        Processes gesture data to move the mouse.

        Args:
            current_gesture_status: A boolean indicating if the gesture is currently active.
            gesture_coords: A tuple containing the (x, y) screen coordinates of the two
                            landmarks, e.g., ((x1, y1), (x2, y2)).
        """
        p1, p2 = gesture_coords
        current_midpoint = self._calculate_midpoint(p1, p2)

        # --- Gesture is Active ---
        if current_gesture_status and current_midpoint:
            # If this is the first frame the gesture is active, set the anchor point.
            if not self.is_gesture_active_prev_frame:
                self.previous_midpoint = current_midpoint
                self.smoothing_buffer.clear()
            else:
                # Calculate the change in position (delta)
                if self.previous_midpoint:
                    dx = current_midpoint[0] - self.previous_midpoint[0]
                    dy = current_midpoint[1] - self.previous_midpoint[1]

                    self.smoothing_buffer.append((dx, dy))

                    if self.smoothing_buffer:
                        avg_dx = sum(item[0] for item in self.smoothing_buffer) / len(self.smoothing_buffer)
                        avg_dy = sum(item[1] for item in self.smoothing_buffer) / len(self.smoothing_buffer)

                        # --- NEW: MOVEMENT DEADZONE LOGIC ---
                        # Calculate the magnitude of the movement vector
                        movement_magnitude = math.sqrt(avg_dx**2 + avg_dy**2)

                        # Only move the mouse if the movement is outside the deadzone
                        if movement_magnitude > self.deadzone_radius:
                            self.mouse.move(avg_dx * self.scale_factor, avg_dy * self.scale_factor)
                        # --- END OF NEW LOGIC ---

                # Update the previous position for the next frame
                self.previous_midpoint = current_midpoint

        # Update the state for the next frame
        self.is_gesture_active_prev_frame = current_gesture_status