import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # or 'yolov8s.pt' for better accuracy

cap = cv2.VideoCapture("Car_Speed_Detection/video_raw/input.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("Car_Speed_Detection/Output_videos/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---- INTERACTIVE LINE DRAWING ----
# ---- DEFAULT LINE POSITIONS ----
default_line1_start = (251, 484)
default_line1_end   = (550, 514)
default_line2_start = (692, 460)
default_line2_end   = (904, 435)

lines = []

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
cv2.namedWindow("Click Lines, press 'q' to complete")
cv2.setMouseCallback("Click Lines", click_event)

# Line drawing loop
while True:
    temp_frame = frame.copy()

    # Draw user lines while clicking
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            color = (0, 255, 255) if i == 0 else (255, 0, 255)
            cv2.line(temp_frame, lines[i], lines[i+1], color, 2)

    cv2.imshow("Click Lines, press 'q' once complete", temp_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

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
            cy = int((y1 + y2) / 2)
            curr_centroid = (cx, cy)
            cv2.circle(frame, curr_centroid, 4, (255, 255, 255), -1)

            if track_id is not None:
                prev_centroid = previous_centroids.get(track_id)
                previous_centroids[track_id] = curr_centroid

                if prev_centroid:
                    if track_id not in counted_ids_1 and crossed_line(prev_centroid, curr_centroid, line1_start, line1_end):
                        counted_ids_1.add(track_id)
                        count1 += 1

                    if track_id not in counted_ids_2 and crossed_line(prev_centroid, curr_centroid, line2_start, line2_end):
                        counted_ids_2.add(track_id)
                        count2 += 1

                cv2.putText(frame, f"{class_name}ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw both lines
    cv2.line(frame, line1_start, line1_end, (0, 255, 255), line_thickness)
    cv2.line(frame, line2_start, line2_end, (255, 0, 255), line_thickness)

    cv2.putText(frame, f"Line 1 Count: {count1}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Line 2 Count: {count2}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("Saved video as output.mp4")
