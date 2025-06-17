import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---

# Path to the input video
video_path = "car_counting/video_raw/highway.mp4"
# Path to the output video
output_path = "car_counting/Output_videos/output_video_gpt.mp4"

# Create YOLOv8 model instance (ensure you have the proper weights file)
# For example: using "yolov8n.pt" or any custom weights trained for vehicle detection
model = YOLO("yolov8n.pt")

# Create DeepSORT tracker instance
# You can adjust parameters such as max_age, n_init, and nn_budget as needed
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=100)

# Initialize the count of vehicles
vehicle_count = 0

# To avoid double counting, we maintain a set of IDs that have already crossed
crossed_ids = set()

# --- DEFINE THE COUNTING LINE ---
# The counting line is defined by two points.
# You can change these coordinates to move the line anywhere on the video.
# For instance, for a bottom left lane, you might use:
line_pt1 = (100, 600)  # Left point of the line (x, y)
line_pt2 = (500, 600)  # Right point of the line (x, y)
# To move the line, simply modify line_pt1 and line_pt2:
# Example: For a line more centered, you could use: (300, 600) and (900, 600)

# Calculate line vector (for later crossing checks)
line_vector = np.array(line_pt2) - np.array(line_pt1)

# --- HELPER FUNCTIONS ---

def get_center(bbox):
    """
    Given a bounding box in the format (x1, y1, x2, y2), return its center (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def is_crossing(line_pt1, line_pt2, pt_prev, pt_curr):
    """
    Check if an object moved from one side of the line to the other.
    Uses the sign of the cross product to determine the relative position of a point to the line.
    """
    def side(point, line_pt1, line_pt2):
        return np.sign((line_pt2[0] - line_pt1[0])*(point[1] - line_pt1[1]) - (line_pt2[1] - line_pt1[1])*(point[0] - line_pt1[0]))
    
    side_prev = side(pt_prev, line_pt1, line_pt2)
    side_curr = side(pt_curr, line_pt1, line_pt2)
    # If the sign changes and neither is 0 (exactly on the line), we consider the line as crossed.
    return side_prev != side_curr and side_prev != 0 and side_curr != 0

# Dictionary to store the last center for each object ID
last_centers = {}

# --- VIDEO PROCESSING ---

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Expected 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Expected 720
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection using YOLOv8
    results = model(frame)[0]
    detections = []  # list to store detections for deep sort

    # Process the detection results
    # YOLO returns bounding boxes with format: [x1, y1, x2, y2] and a confidence score
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        # Optionally filter by class: check if the detection is a car (class id may differ, adjust as needed)
        # For example, if your model is trained with COCO:
        # Cars are often class 2 for "car"
        # if int(cls) != 2:
        #     continue
        detections.append(([int(x1), int(y1), int(x2), int(y2)], conf, "car"))

    # Update tracker with current detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process the tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # returns bounding box in [left, top, right, bottom]
        center = get_center(ltrb)
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (int(ltrb[0]), int(ltrb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.circle(frame, center, 4, (255, 0, 0), -1)

        # If we have seen this ID before, check if it crossed the line between the previous and current frame.
        if track_id in last_centers:
            if is_crossing(line_pt1, line_pt2, last_centers[track_id], center):
                # Check if this ID is already counted
                if track_id not in crossed_ids:
                    vehicle_count += 1
                    crossed_ids.add(track_id)
        # Store current center for next frame
        last_centers[track_id] = center

    # Draw the counting line on the frame:
    cv2.line(frame, line_pt1, line_pt2, (0, 0, 255), 2)
    # Add a text label with the current vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # Display the frame (optional)
    cv2.imshow("Vehicle Counting", frame)
    out.write(frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()