import time
import math
from Quartz import CGDisplayBounds, CGWarpMouseCursorPosition, CGPoint

# --- Configuration ---
MOVE_DURATION = 10  # Total duration of the movement in seconds

# --- Script Start ---
print("Starting in 3 seconds... Get ready!")
time.sleep(3)

# 1. Get the bounds of the main display using Core Graphics
main_display_bounds = CGDisplayBounds(0) # 0 is the ID for the main display
screenWidth = main_display_bounds.size.width
screenHeight = main_display_bounds.size.height

# Define start (bottom-left) and end (top-right) coordinates
start_x, start_y = 0, screenHeight
end_x, end_y = screenWidth, 0

# Instantly move to the starting position using the native function
# The function expects a CGPoint object, which is essentially a (x, y) tuple
start_point = CGPoint(start_x, start_y)
CGWarpMouseCursorPosition(start_point)
print("Cursor at bottom-left. Starting native macOS glide... 🚀")

# 2. The Animation Loop (now with native calls)
start_time = time.time()
while True:
    elapsed_time = time.time() - start_time
    progress = min(1.0, elapsed_time / MOVE_DURATION)
    
    # Apply the same smooth easing function for acceleration/deceleration
    eased_progress = -(math.cos(math.pi * progress) - 1) / 2

    # Calculate the new interpolated X and Y coordinates
    current_x = start_x + (end_x - start_x) * eased_progress
    current_y = start_y + (end_y - start_y) * eased_progress

    # 3. Move the mouse using the direct Core Graphics call
    new_point = CGPoint(current_x, current_y)
    CGWarpMouseCursorPosition(new_point)

    if progress >= 1.0:
        break

# 4. Final positioning to ensure it ends perfectly
end_point = CGPoint(end_x, end_y)
CGWarpMouseCursorPosition(end_point)
print("Done! This should feel buttery-smooth. ✨")