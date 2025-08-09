import time
import Quartz

def smooth_move(x1, y1, x2, y2, duration=10, steps=600):
    dx = (x2 - x1) / steps
    dy = (y2 - y1) / steps
    delay = duration / steps

    for i in range(steps + 1):
        Quartz.CGWarpMouseCursorPosition((x1 + dx * i, y1 + dy * i))
        Quartz.CGAssociateMouseAndMouseCursorPosition(True)
        time.sleep(delay)

if __name__ == "__main__":
    # Get screen size
    main_display = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    screen_width = int(main_display.size.width)
    screen_height = int(main_display.size.height)

    start_x, start_y = 0, screen_height - 1
    end_x, end_y = screen_width - 1, 0

    smooth_move(start_x, start_y, end_x, end_y, duration=10, steps=1200)
