import cv2
import torch # Although not directly used, often needed by YOLO/ultralytics
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Configuration ---
VIDEO_PATH = 'car_counting/video_raw/highway.mp4' # Replace with your video file path
OUTPUT_VIDEO_PATH = 'car_counting/Output_videos/output_video_gem.mp4'
YOLO_MODEL_PATH = 'yolov8n.pt' # Or yolov8s.pt, yolov8m.pt etc. depending on desired accuracy/speed
CONFIDENCE_THRESHOLD = 0.5 # Minimum detection confidence
IOU_THRESHOLD = 0.5 # NMS threshold for DeepSORT
MAX_AGE = 50 # Max frames to keep a track without detection
N_INIT = 3 # Min number of detections to start a track

# Define the Class ID for 'car' in the COCO dataset (YOLOv8 default)
# Check your specific model's documentation if unsure. Usually: 2=car, 7=truck
CAR_CLASS_ID = 2

# --- Line Configuration ---
# Define the coordinates for the counting line.
# To move the line: Change these coordinates.
# Example: Horizontal line in the bottom-left quadrant
# LINE_START = (0, 500) # Starting point (x1, y1) - Left edge, 500 pixels down
# LINE_END = (640, 500) # Ending point (x2, y1) - Halfway width, same height

# Example 2: Vertical line in the center-left
# LINE_START = (320, 0) # Starting point (x1, y1) - Quarter width, top edge
# LINE_END = (320, 720) # Ending point (x1, y2) - Quarter width, bottom edge

# ** INITIAL LINE ** (Bottom Left Quadrant - Horizontal)
# Adjust x2 (640) and y1 (500) to move the line
# Make x2 smaller/larger to shorten/lengthen the line horizontally.
# Make y1 smaller/larger to move the line up/down.
LINE_START = (100, 600) # (x1, y1) - Start point slightly offset from left edge
LINE_END = (500, 600)  # (x2, y1) - End point extending further right, same height

# --- Initialization ---
print("Loading YOLOv8 model...")
model = YOLO(YOLO_MODEL_PATH)
# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model.to(device)

print("Initializing DeepSORT tracker...")
# Note: Adjust parameters based on video characteristics and desired performance
tracker = DeepSort(max_iou_distance=IOU_THRESHOLD,
                   max_age=MAX_AGE,
                   n_init=N_INIT,
                   nms_max_overlap=1.0, # Using default, can be tuned
                   max_cosine_distance=0.3, # Default, appearance threshold
                   nn_budget=None, # Default uses unlimited budget
                   override_track_class=None, # We handle class filtering ourselves
                   embedder="mobilenet", # Or "osnet", "resnet", etc.
                   half=True if device == 'cuda' else False, # Use FP16 on GPU
                   bgr=True, # OpenCV uses BGR format
                   embedder_gpu=True if device == 'cuda' else False,
                   embedder_model_name=None,
                   embedder_wts=None,
                   polygon=False,
                   today=None)

print("Opening video file...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID'
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# --- Processing Variables ---
vehicle_counter = 0
counted_track_ids = set() # Keep track of IDs that have already been counted

# --- Main Processing Loop ---
frame_count = 0
print("Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or error reading frame.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # 1. YOLOv8 Detection
    # results is a list, usually with one element for the image/frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[CAR_CLASS_ID], device=device, verbose=False)

    # Prepare detections for DeepSORT
    detections_for_deepsort = []
    # results[0].boxes contains bounding boxes, confidences, classes for the first image
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates (top-left, bottom-right)
        confidence = float(box.conf[0])       # Detection confidence
        class_id = int(box.cls[0])            # Class ID

        # Format for DeepSORT: [ [left, top, w, h], confidence, detection_class_id ]
        w = x2 - x1
        h = y2 - y1
        detection_data = [ [x1, y1, w, h], confidence, class_id ]
        detections_for_deepsort.append(detection_data)

    # 2. DeepSORT Tracking
    # Update tracker with current detections
    # The 'frame' is often needed for appearance feature extraction
    tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)

    # 3. Process Tracks and Count Vehicles
    valid_tracks = []
    for track in tracks:
        # Skip tracks that are not confirmed (still tentative)
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb() # Get bounding box in [left, top, right, bottom] format
        x1, y1, x2, y2 = map(int, ltrb)

        # Calculate bottom center point of the bounding box
        # This point is used to check if the car crosses the line
        center_x = (x1 + x2) // 2
        bottom_y = y2

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Counting Logic ---
        # Check if the bottom center point crosses the line
        # Basic check: if the point is close to the line y-coordinate
        # A more robust check would involve checking direction or storing previous positions
        line_y = LINE_START[1] # Assuming a horizontal line
        is_crossing = False

        # Simple Horizontal Line Crossing Check (Assumes downward or upward motion across the line)
        # Adjust tolerance (e.g., +/- 5 pixels) if needed
        tolerance = 10 # Number of pixels around the line to trigger check
        if abs(bottom_y - line_y) < tolerance:
             # Check if this track ID has already been counted
            if track_id not in counted_track_ids:
                vehicle_counter += 1
                counted_track_ids.add(track_id)
                print(f"Vehicle counted! ID: {track_id}, Total Count: {vehicle_counter}")
                # Optional: Change color when counted
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box when counted


        # Note: For a vertical line, you would check the center_x against LINE_START[0]
        # Note: For diagonal lines, you need a point-line distance calculation.

    # 4. Draw Counting Line and Counter Display
    # Draw the counting line on the frame
    cv2.line(frame, LINE_START, LINE_END, (255, 0, 0), 3) # Blue line, thickness 3

    # Display the vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_counter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 5. Write Output Frame
    out.write(frame)

    # 6. Display Frame (Optional)
    # cv2.imshow("Vehicle Counting", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# --- Cleanup ---
print("Finished processing.")
print(f"Total vehicles counted: {vehicle_counter}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")