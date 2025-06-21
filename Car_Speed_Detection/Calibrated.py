import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- Global Variables for Calibration ---
calibration_points = []
pixels_per_meter = None
KNOWN_DISTANCE_METERS = 13.0

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse clicks to get calibration points."""
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Point {len(calibration_points)} selected: ({x}, {y})")

def main():
    global pixels_per_meter, calibration_points

    # --- Configuration ---
    video_path = 'Car_Speed_Detection/video_raw/input.mp4'
    output_video_path = 'Car_Speed_Detection/Output_videos/speed_output.mp4'
    
    # --- Default Calibration Points ---
    # These are fallback points if the user doesn't provide them.
    # NOTE: These are placeholders and MUST be adjusted for your specific video.
    # Format: [dist_p1, dist_p2, line_p1, line_p2]
    default_points = [(353, 463), (202, 574), (269, 469), (569, 491)]
    # --- Component 1: Interactive 4-Point Calibration ---
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
    print("Please click 4 points:")
    print("1. Two points to define the known distance (13m).")
    print("2. Two more points to define the speed activation line.")
    print("After selecting points, press 'c' to confirm, or 'Esc' to exit.")

    while True:
        frame_copy = first_frame.copy()
        if len(calibration_points) > 0:
            for i, point in enumerate(calibration_points):
                cv2.circle(frame_copy, point, 7, (0, 0, 255), -1)
                cv2.putText(frame_copy, str(i+1), (point[0]+10, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if len(calibration_points) > 1:
            cv2.line(frame_copy, calibration_points[0], calibration_points[1], (0, 255, 0), 2)
        if len(calibration_points) > 3:
            cv2.line(frame_copy, calibration_points[2], calibration_points[3], (255, 0, 0), 2)

        cv2.imshow("Calibration", frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return
        if key == ord('c'):
            if len(calibration_points) < 4:
                print("4 points not detected. Going with default values.")
                calibration_points = default_points
            break

    cv2.destroyWindow("Calibration")
    cv2.waitKey(1) # allow window to close properly
    distance_points = calibration_points[:2]
    activation_line_points = calibration_points[2:]
    
    pixel_distance = np.linalg.norm(np.array(distance_points[0]) - np.array(distance_points[1]))
    pixels_per_meter = pixel_distance / KNOWN_DISTANCE_METERS
    print(f"Calibration complete. Pixels per meter: {pixels_per_meter:.2f}")

    # --- Initialize Components ---
    model = YOLO('yolov8s.pt')
    vehicle_class_ids = [2, 3, 5, 7] # car, motorbike, bus, truck
    class_names = model.names

    track_history = defaultdict(list)
    activated_tracks = set()
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    frame_number = 0
    overlay_update_frequency = int(video_fps)
    displayable_overlay_info = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Processing video...")

    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        results = model.track(frame, persist=True, classes=vehicle_class_ids, verbose=False)

        active_tracks_info = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                centroid_x = int((box[0] + box[2]) / 2)
                centroid_y = int (box[3])
                centroid = (centroid_x, centroid_y)
                
                # Activation Line Logic
                if track_id not in activated_tracks:
                    p1, p2 = activation_line_points
                    cross_product = (p2[0] - p1[0]) * (centroid_y - p1[1]) - (p2[1] - p1[1]) * (centroid_x - p1[0])
                    if cross_product > 0:
                        activated_tracks.add(track_id)

                if track_id in activated_tracks:
                    history = track_history[track_id]
                    history.append(centroid)

                    speed_kph = 0
                    if len(history) > 1:
                        pixel_dist = np.linalg.norm(np.array(history[-1]) - np.array(history[-2]))
                        meter_dist = pixel_dist / pixels_per_meter
                        speed_kph = (meter_dist * video_fps) * 3.6

                    class_name = class_names[cls]
                    active_tracks_info.append(f"{class_name} {track_id}: {int(speed_kph)} km/h")
                    
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.circle(frame, centroid, 5, (255, 255, 255), -1)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (int(box[0]), int(box[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw calibration and activation lines
        cv2.line(frame, distance_points[0], distance_points[1], (0, 255, 255), 2)
        cv2.line(frame, activation_line_points[0], activation_line_points[1], (255, 0, 255), 2)

        # Update and draw the top-right speed overlay less frequently
        if frame_number % overlay_update_frequency == 0:
            displayable_overlay_info = sorted(active_tracks_info)
        
        overlay_y_start = 40
        for i, info_text in enumerate(displayable_overlay_info):
            cv2.putText(frame, info_text, (frame_width - 300, overlay_y_start + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        video_writer.write(frame)

    # --- Cleanup ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to '{output_video_path}'")

if __name__ == "__main__":
    main()
