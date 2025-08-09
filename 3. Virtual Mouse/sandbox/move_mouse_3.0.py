import pyautogui
pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Bottom-left starting point
start_x, start_y = 0, screen_height - 1

# Top-right target point
end_x, end_y = screen_width - 1, 0

# Move instantly to bottom-left
pyautogui.moveTo(start_x, start_y)

# Smoothly move to top-right over 10 seconds
pyautogui.moveTo(end_x, end_y, duration=10,tween=pyautogui.easeInOutQuad)
