import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import csv

# --- Global Variables for Calibration ---
calibration_points = []
pixels_per_meter = None
KNOWN_DISTANCE_METERS = 13.0

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse clicks to get calibration points."""
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            print(f"Point {len(calibration_points)} selected: ({x}, {y})")

def main():
    global pixels_per_meter

    # --- Configuration ---
    video_path = 'Car_Speed_Detection/video_raw/input.mp4'
    output_video_path = 'speed_output.mp4'
    output_csv_path = 'speed_data.csv'

    # --- Component 1: Interactive 2-Point Calibration ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    print("Please click on two points on the road to define your known distance.")
    print(f"The assumed real-world distance is {KNOWN_DISTANCE_METERS} meters.")
    print("After selecting two points, press 'c' to confirm, or 'Esc' to exit.")

    while True:
        frame_copy = first_frame.copy()
        if len(calibration_points) > 0:
            cv2.circle(frame_copy, calibration_points[0], 7, (0, 0, 255), -1)
        if len(calibration_points) > 1:
            cv2.circle(frame_copy, calibration_points[1], 7, (0, 0, 255), -1)
            cv2.line(frame_copy, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
        
        cv2.imshow("Calibration", frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return
        if key == ord('c') and len(calibration_points) == 2:
            break

    cv2.destroyAllWindows()

    p1, p2 = calibration_points
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    pixels_per_meter = pixel_distance / KNOWN_DISTANCE_METERS
    print(f"Calibration complete. Pixels per meter: {pixels_per_meter:.2f}")

    # --- Initialize YOLO, Data Structures, and Video Writer ---
    model = YOLO('yolov8n.pt')
    vehicle_class_ids = [2, 3, 5, 7] # car, motorbike, bus, truck
    class_names = model.names

    track_history = defaultdict(list)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    frame_number = 0
    all_speed_data = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Processing video to track vehicles, calculate speed, and generate output video...")

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        # --- Combined Detection and Tracking ---
        results = model.track(frame, persist=True, classes=vehicle_class_ids, verbose=False)

        active_tracks_info = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # --- Speed Calculation ---
                centroid_x = int((box[0] + box[2]) / 2)
                centroid_y = int((box[1] + box[3]) / 2)
                
                history = track_history[track_id]
                history.append((centroid_x, centroid_y))

                speed_kph = 0
                if len(history) > 1:
                    pixel_dist = np.linalg.norm(np.array(history[-1]) - np.array(history[-2]))
                    meter_dist = pixel_dist / pixels_per_meter
                    speed_kph = (meter_dist * video_fps) * 3.6
                    
                    all_speed_data.append({
                        'frame_number': frame_number,
                        'track_id': track_id,
                        'class_name': class_names[cls],
                        'speed_kph': round(speed_kph, 2),
                        'pixel_distance_per_frame': round(pixel_dist, 2)
                    })
                
                # Store info for the top-right overlay
                class_name = class_names[cls]
                active_tracks_info.append(f"{class_name} {track_id}: {int(speed_kph)} km/h")
                
                # --- On-Frame Visualizations ---
                # Bounding Box (Red)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                # Centroid (White)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 255), -1)

        # Draw the calibration line
        cv2.line(frame, calibration_points[0], calibration_points[1], (0, 255, 255), 2)

        # Draw the top-right speed overlay
        overlay_y_start = 40
        for i, info_text in enumerate(active_tracks_info):
            cv2.putText(frame, info_text, (frame_width - 250, overlay_y_start + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write the frame to the output video
        video_writer.write(frame)

    # --- Cleanup and CSV Writing ---
    cap.release()
    video_writer.release()
    print("Video processing complete.")

    if all_speed_data:
        print(f"Writing {len(all_speed_data)} records to '{output_csv_path}'...")
        headers = ['frame_number', 'track_id', 'class_name', 'speed_kph', 'pixel_distance_per_frame']
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_speed_data)
        print("CSV file successfully created.")
    else:
        print("No vehicle speed data was generated.")

if __name__ == "__main__":
    main()
