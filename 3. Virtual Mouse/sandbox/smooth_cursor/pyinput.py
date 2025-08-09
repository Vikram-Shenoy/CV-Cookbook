import time
import math
from pynput.mouse import Controller
import pyautogui # Still using this just to get screen size easily

# --- Configuration ---
MOVE_DURATION = 10  # Total duration of the movement in seconds

# --- Initialize ---
mouse = Controller()

# --- Script Start ---
print("Starting in 1 seconds... Get ready!")
time.sleep(1)

# 1. Get screen dimensions and define start/end points
# We use pyautogui here because its size() function is reliable and simple
screenWidth, screenHeight = pyautogui.size() 
start_x, start_y = 0, screenHeight - 1
end_x, end_y = screenWidth - 1, 0

# Instantly move to the starting position using pynput
mouse.position = (start_x, start_y)
print("Cursor at bottom-left. Starting high-frequency glide with pynput...")

# 2. The Animation Loop
start_time = time.time()
while True:
    elapsed_time = time.time() - start_time
    progress = min(1.0, elapsed_time / MOVE_DURATION) # Ensure progress doesn't exceed 1.0

    # Apply the same smooth easing function
    eased_progress = -(math.cos(math.pi * progress) - 1) / 2

    # Calculate the new interpolated X and Y coordinates
    current_x = start_x + (end_x - start_x) * eased_progress
    current_y = start_y + (end_y - start_y) * eased_progress
    
    # 3. Move the mouse to the calculated position using pynput
    mouse.position = (current_x, current_y)

    if progress >= 1.0:
        break

# 4. Final positioning to ensure it ends perfectly
mouse.position = (end_x, end_y)
print("Done! ✨")