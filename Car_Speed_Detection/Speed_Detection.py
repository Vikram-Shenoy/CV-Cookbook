import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime

entry_times = {}  # track_id → entry timestamp
exit_records = []  # list of dicts: {id, entry_time, exit_time}


model = YOLO("yolov8s.pt")  # or 'yolov8m.pt' for better accuracy

cap = cv2.VideoCapture("Car_Speed_Detection/video_raw/input.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("Car_Speed_Detection/Output_videos/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---- INTERACTIVE LINE DRAWING ----
# ---- DEFAULT LINE POSITIONS ----
default_line1_start = (287, 459)
default_line1_end   = (578, 472)
default_line2_start = (133, 553)
default_line2_end   = (545, 597)

lines = []
speed_display = []  # to store speed display messages
dist = 13  # distance in meters between the two lines
mouse_position = (0, 0)

def live_mouse_pos(event, x, y, flags, param):
    global mouse_position
    mouse_position = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and len(lines) < 4:
        lines.append((x, y))
        print(f"Clicked: ({x}, {y})")
import cv2

def put_text_with_simple_blur(img, text, org, font, font_scale, color, thickness=1, pad=1):
    # Get text size
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org

    # Define ROI coordinates with padding
    x1, y1 = max(x - pad, 0), max(y - h - baseline - pad, 0)
    x2, y2 = x + w + pad, y + baseline + pad

    # Extract ROI and apply blur
    roi = img[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

    # Put blurred ROI back
    img[y1:y2, x1:x2] = blurred_roi

    # Draw text on top
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)



# Deprecated click_event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(lines) < 4:
        lines.append((x, y))
        print(f"Clicked: ({x}, {y})")

# Grab the first frame
ret, frame = cap.read()
if not ret:
    print("Error reading video")
    exit()

print("Click 2 points for Line 1, then 2 points for Line 2. Press 'q' to confirm and continue.")
cv2.namedWindow("Click Lines")
cv2.setMouseCallback("Click Lines", live_mouse_pos)

# Line drawing loop
while True:
    temp_frame = frame.copy()
    num_points = len(lines)
    # Draw user lines while clicking
    for i in range(0, num_points, 2):
        if i + 1 < num_points:
            color = (0, 255, 255) if i == 0 else (255, 0, 255)
            cv2.line(temp_frame, lines[i], lines[i+1], color, 2)

    if num_points % 2 == 1:
        color = (0, 255, 255) if num_points == 1 else (255, 0, 255)
        cv2.line(temp_frame, lines[-1], mouse_position, color, 1)


    cv2.imshow("Click Lines", temp_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 27:  # ESC key pressed
        lines.clear()
        print("Resetting clicked points... start again.")
        flash = frame.copy()
        overlay = flash.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.1  # transparency
        cv2.addWeighted(overlay, alpha, flash, 1 - alpha, 0, flash)
        cv2.imshow("Click Lines", flash)
        cv2.waitKey(100)

# Ensure windows are closed BEFORE continuing
cv2.setMouseCallback("Click Lines", lambda *args: None)  # remove callback
cv2.destroyWindow("Click Lines")
cv2.waitKey(1)  # allow window to close properly

# Use clicked or fallback to default
if len(lines) == 4:
    print("Using user-defined line coordinates.")
    line1_start, line1_end = lines[0], lines[1]
    line2_start, line2_end = lines[2], lines[3]
else:
    print("Not enough clicks detected. Using default line coordinates.")
    line1_start, line1_end = default_line1_start, default_line1_end
    line2_start, line2_end = default_line2_start, default_line2_end

# Reset video to frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
line_thickness = 3

# Detection filtering
min_box_area = 3000
count1, count2 = 0, 0
counted_ids_1 = set()
counted_ids_2 = set()
previous_centroids = {}

# --- Helper: Check if a line was crossed ---
def crossed_line(p1, p2, line_start, line_end):
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and ccw(p1, p2, line_start) != ccw(p1, p2, line_end)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video after frame selection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            if class_name not in ['car', 'truck', 'bus', 'motorbike']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < min_box_area:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            track_id = int(box.id.item()) if box.id is not None else None
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            curr_centroid = (cx, cy)
            cv2.circle(frame, curr_centroid, 4, (255, 255, 255), -1)

            if track_id is not None:
                prev_centroid = previous_centroids.get(track_id)
                previous_centroids[track_id] = curr_centroid

                if prev_centroid:
                    # ENTRY
                    if track_id not in entry_times and crossed_line(prev_centroid, curr_centroid, line1_start, line1_end):
                        entry_times[track_id] = datetime.now()
                    
                    # EXIT
                    elif track_id in entry_times and crossed_line(prev_centroid, curr_centroid, line2_start, line2_end):
                        exit_time = datetime.now()
                        entry_time = entry_times.pop(track_id)
                        time_diff = (exit_time - entry_time).total_seconds()
                        speed_mps = dist / time_diff
                        speed_kmph = speed_mps * 3.6  # convert to km/h

                        speed_display.append(f"Vehicle {track_id}: {speed_kmph:.1f} km/h")
                        exit_records.append({
                            'Vehicle_id': track_id,
                            'Entry_time': entry_time.strftime('%H:%M:%S.%f'),
                            'Exit_time': exit_time.strftime('%H:%M:%S.%f'),
                        })
                cv2.putText(frame, f"{class_name} ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw both lines
    cv2.line(frame, line1_start, line1_end, (0, 255, 255), line_thickness)
    cv2.line(frame, line2_start, line2_end, (255, 0, 255), line_thickness)
    # Overlay speed display
    for i, text in enumerate(speed_display[-15:]):  # show last 10 vehicles max
        y = 20 + i * 30
        put_text_with_simple_blur(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    out.write(frame)

with open('vehicle_timings.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Vehicle_id', 'Entry_time', 'Exit_time'])
    writer.writeheader()
    writer.writerows(exit_records)

print("Saved vehicle timings to vehicle_timings.csv")

cap.release()
out.release()
print("Saved video as output.mp4")
