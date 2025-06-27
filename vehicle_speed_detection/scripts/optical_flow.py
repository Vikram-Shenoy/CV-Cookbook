import cv2
import numpy as np
import csv

# --- Global Variables for Calibration and Setup ---
roi_points = []
pixels_per_meter = None
KNOWN_DISTANCE_METERS = 10.0

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse clicks to get ROI points."""
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            roi_points.append((x, y))
            print(f"ROI Point {len(roi_points)} selected: ({x}, {y})")

def main():
    global pixels_per_meter, roi_points

    # --- Configuration ---
    video_path = "vehicle_speed_detection/videos/input/highway_clipped.mp4"
    output_video_path = 'vehicle_speed_detection/videos/output/sparse_flow_output.mp4'
    output_csv_path = 'vehicle_speed_detection/data/sparse_flow_data.csv'
    
    # --- Default ROI Points ---
    # NOTE: These are placeholders and MUST be adjusted for your specific video.
    default_roi = [(459, 469),(394, 545),(532, 567),(562, 479)]

    # --- Component 1: Interactive ROI and Calibration Setup ---
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
    print("Please click 4 points to define the rectangular ROI.")
    print("The first two points will also be used for distance calibration (13m).")
    print("After selecting points, press 'c' to confirm, or 'Esc' to exit.")

    while True:
        frame_copy = first_frame.copy()
        if len(roi_points) > 0:
            for i, point in enumerate(roi_points):
                cv2.circle(frame_copy, point, 7, (0, 0, 255), -1)
        if len(roi_points) == 4:
             cv2.polylines(frame_copy, [np.array(roi_points, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.imshow("Calibration", frame_copy)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return
        if key == ord('c'):
            if len(roi_points) < 4:
                print("4 points not detected. Using default ROI values.")
                roi_points = default_roi
            break

    cv2.destroyWindow("Calibration")
    cv2.waitKey(1)

    distance_points = roi_points[:2]
    pixel_distance = np.linalg.norm(np.array(distance_points[0]) - np.array(distance_points[1]))
    pixels_per_meter = pixel_distance / KNOWN_DISTANCE_METERS
    print(f"Calibration complete. Pixels per meter: {pixels_per_meter:.2f}")
    
    # --- Initialize for Optical Flow ---
    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Feature detection parameters
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Prepare for main loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for the ROI
    roi_mask = np.zeros_like(old_gray)
    cv2.fillPoly(roi_mask, [np.array(roi_points, np.int32)], 255)
    
    # Find initial features to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

    # Setup video writer
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    all_frame_data = []
    frame_number = 0
    print("Processing video...")

    # --- Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate sparse optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            good_new, good_old = [], []

        speed_kph = 0
        avg_pixel_speed = 0
        tracked_points_count = len(good_new)
        
        if tracked_points_count > 0:
            # Calculate the average displacement of all tracked features
            displacements = np.linalg.norm(good_new - good_old, axis=1)
            avg_pixel_speed = np.mean(displacements)
            
            # Convert to absolute speed
            avg_meter_speed = avg_pixel_speed / pixels_per_meter
            speed_ms = avg_meter_speed * video_fps
            speed_kph = speed_ms * 3.6

        # Log data for this frame
        all_frame_data.append({
            'frame_number': frame_number,
            'tracked_points_count': tracked_points_count,
            'avg_pixel_speed_per_frame': round(avg_pixel_speed, 2),
            'speed_kph': round(speed_kph, 2)
        })

        # --- Visualization ---
        # Draw the shaded ROI
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(roi_points, np.int32)], (0, 200, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [np.array(roi_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw the feature points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Draw the region speed text
        cv2.putText(frame, f"Region Speed: {int(speed_kph)} km/h", 
                    (frame_width - 500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 2)
        
        video_writer.write(frame)

        # --- Update state for next frame ---
        old_gray = frame_gray.copy()
        
        # Re-detect if it's time for a refresh OR if we lost all our points.
        if frame_number % 30 == 0 or len(good_new) == 0:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)
        else:
            p0 = good_new.reshape(-1, 1, 2)
        
        # If p0 is None (no features found), we can't continue.
        if p0 is None:
            print("Could not find new features to track. Stopping.")
            break

    # --- Cleanup and CSV Writing ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to '{output_video_path}'")

    if all_frame_data:
        print(f"Writing data to '{output_csv_path}'...")
        headers = ['frame_number', 'tracked_points_count', 'avg_pixel_speed_per_frame', 'speed_kph']
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_frame_data)
        print("CSV file successfully created.")

if __name__ == "__main__":
    main()
