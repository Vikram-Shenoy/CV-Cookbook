import cv2
import numpy as np
import csv
import sys
import os
# This adds the parent directory (vehicle_speed_detection) to the Python path
# so it can find the 'utils' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.zone_creation import create_zones_from_video
except ImportError:
    print("Error: Could not import 'create_zones_from_video' from 'utils/zone_creation.py'.")
    print("Please ensure the file exists and is in the correct directory.")
    exit(1)

def eight_pts_from_six(six_points):
        """ 
        Assuming the points are ordered as follows:
                0-2-4
                | | |
                1-3-5
        LeftMost edge, Middle edge, RightMost edge
        """
        roi_points_1 = [six_points[2], six_points[3], six_points[5], six_points[4]]
        roi_points_2 = [six_points[2], six_points[3], six_points[1], six_points[0]]
        return roi_points_1, roi_points_2

def main():
    # --- Configuration ---
    video_path = "vehicle_speed_detection/videos/input/highway_clipped.mp4"
    output_video_path = 'vehicle_speed_detection/videos/output/sparse_flow_output.mp4'
    output_csv_path_1 = 'vehicle_speed_detection/data/sparse_flow_data_zone_1.csv'
    output_csv_path_2 = 'vehicle_speed_detection/data/sparse_flow_data_zone_2.csv'
    KNOWN_DISTANCE_METERS = 10.0


    six_points = create_zones_from_video(video_path)
    if not six_points or len(six_points) != 6:
        print("Error: The zone creation function did not return 6 points. Using default zones.")
        six_points = [(289,466), (224,542), (439,466), (374,542), (589,466), (524,542)]

    # Getting eight points from the six points
    roi_points_1, roi_points_2 = eight_pts_from_six(six_points)

    distance_points_1 = roi_points_1[:2]
    pixel_distance_1 = np.linalg.norm(np.array(distance_points_1[0]) - np.array(distance_points_1[1]))
    pixels_per_meter_1 = pixel_distance_1 / KNOWN_DISTANCE_METERS
    print(f"Zone 1 Calibration complete. Pixels per meter: {pixels_per_meter_1:.2f}")

    distance_points_2 = roi_points_2[:2]
    pixel_distance_2 = np.linalg.norm(np.array(distance_points_2[0]) - np.array(distance_points_2[1]))
    pixels_per_meter_2 = pixel_distance_2 / KNOWN_DISTANCE_METERS
    print(f"Zone 2 Calibration complete. Pixels per meter: {pixels_per_meter_2:.2f}")

    # --- Initialize for Optical Flow ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Create two masks and find initial features for both zones
    roi_mask_1 = np.zeros_like(old_gray)
    cv2.fillPoly(roi_mask_1, [np.array(roi_points_1, np.int32)], 255)
    p0_1 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_1, **feature_params)

    roi_mask_2 = np.zeros_like(old_gray)
    cv2.fillPoly(roi_mask_2, [np.array(roi_points_2, np.int32)], 255)
    p0_2 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_2, **feature_params)

    # Setup video writer
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    all_frame_data_1 = []
    all_frame_data_2 = []
    frame_number = 0
    print("Processing video...")

    # --- Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Process Zone 1 ---
        good_new_1 = []
        speed_kph_1 = 0
        if p0_1 is not None and len(p0_1) > 0:
            p1_1, st_1, err_1 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_1, None, **lk_params)
            good_new_1, good_old_1 = (p1_1[st_1 == 1], p0_1[st_1 == 1]) if p1_1 is not None else ([], [])
            
            if len(good_new_1) > 0:
                displacements = np.linalg.norm(good_new_1 - good_old_1, axis=1)
                avg_pixel_speed = np.mean(displacements)
                speed_ms = (avg_pixel_speed / pixels_per_meter_1) * video_fps
                speed_kph_1 = speed_ms * 3.6
        
        all_frame_data_1.append({
            'frame_number': frame_number, 'tracked_points_count': len(good_new_1),
            'avg_pixel_speed_per_frame': round(np.mean(np.linalg.norm(good_new_1 - good_old_1, axis=1)) if len(good_new_1) > 0 else 0, 2),
            'speed_kph': round(speed_kph_1, 2)
        })

        # --- Process Zone 2 ---
        good_new_2 = []
        speed_kph_2 = 0
        if p0_2 is not None and len(p0_2) > 0:
            p1_2, st_2, err_2 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_2, None, **lk_params)
            good_new_2, good_old_2 = (p1_2[st_2 == 1], p0_2[st_2 == 1]) if p1_2 is not None else ([], [])
            
            if len(good_new_2) > 0:
                displacements = np.linalg.norm(good_new_2 - good_old_2, axis=1)
                avg_pixel_speed = np.mean(displacements)
                speed_ms = (avg_pixel_speed / pixels_per_meter_2) * video_fps
                speed_kph_2 = speed_ms * 3.6

        all_frame_data_2.append({
            'frame_number': frame_number, 'tracked_points_count': len(good_new_2),
            'avg_pixel_speed_per_frame': round(np.mean(np.linalg.norm(good_new_2 - good_old_2, axis=1)) if len(good_new_2) > 0 else 0, 2),
            'speed_kph': round(speed_kph_2, 2)
        })

        # --- Visualization ---
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(roi_points_1, np.int32)], (0, 200, 0))
        cv2.fillPoly(overlay, [np.array(roi_points_2, np.int32)], (200, 0, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [np.array(roi_points_1, np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [np.array(roi_points_2, np.int32)], isClosed=True, color=(255, 100, 100), thickness=1)

        for new in good_new_1:
            a, b = new.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
        for new in good_new_2:
            a, b = new.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 3, (255, 255, 0), -1)

        cv2.putText(frame, f"Zone 1 Speed: {int(speed_kph_1)} km/h", 
                    (frame_width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Zone 2 Speed: {int(speed_kph_2)} km/h", 
                    (frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2,lineType=cv2.LINE_AA)
        
        video_writer.write(frame)

        # --- Update state for next frame ---
        old_gray = frame_gray.copy()
        
        if frame_number % 30 == 0:
            p0_1 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_1, **feature_params)
            p0_2 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_2, **feature_params)
        else:
            p0_1 = good_new_1.reshape(-1, 1, 2) if len(good_new_1) > 0 else None
            p0_2 = good_new_2.reshape(-1, 1, 2) if len(good_new_2) > 0 else None
        
        if p0_1 is None: p0_1 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_1, **feature_params)
        if p0_2 is None: p0_2 = cv2.goodFeaturesToTrack(old_gray, mask=roi_mask_2, **feature_params)

        if p0_1 is None and p0_2 is None:
            print(f"Frame {frame_number}: No features found to track. Skipping.")
            continue

    # --- Cleanup and CSV Writing ---
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to '{output_video_path}'")
    
    if all_frame_data_1:
        print(f"Writing data to '{output_csv_path_1}'...")
        headers = ['frame_number', 'tracked_points_count', 'avg_pixel_speed_per_frame', 'speed_kph']
        with open(output_csv_path_1, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_frame_data_1)
        print("Zone 1 CSV file successfully created.")
    
    if all_frame_data_2:
        print(f"Writing data to '{output_csv_path_2}'...")
        headers = ['frame_number', 'tracked_points_count', 'avg_pixel_speed_per_frame', 'speed_kph']
        with open(output_csv_path_2, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_frame_data_2)
        print("Zone 2 CSV file successfully created.")

if __name__ == "__main__":
    main()
