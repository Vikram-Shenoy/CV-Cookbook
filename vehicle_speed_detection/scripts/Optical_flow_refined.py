import cv2
import numpy as np
import csv
import sys
import os
from dataclasses import dataclass, field

# This adds the parent directory to the Python path to find the 'utils' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils.zone_creation import create_zones_from_video
except ImportError:
    print("Error: Could not import 'create_zones_from_video' from 'utils/zone_creation.py'.")
    print("Please ensure the file exists and is in the correct directory.")
    exit(1)

@dataclass
class AppConfig:
    """Centralized configuration for the application."""
    video_path: str = "vehicle_speed_detection/videos/input/highway_clipped.mp4"
    output_video_path: str = 'vehicle_speed_detection/videos/output/sparse_flow_output.mp4'
    output_csv_prefix: str = 'vehicle_speed_detection/data/sparse_flow_data_zone'
    known_distance_meters: float = 10.0
    
    lk_params: dict = field(default_factory=lambda: dict(
        winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    ))
    feature_params: dict = field(default_factory=lambda: dict(
        maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
    ))

class ZoneProcessor:
    """
    Encapsulates all data and processing logic for a single detection zone.
    """
    def __init__(self, zone_id: int, roi_points: list, config: AppConfig, display_color: tuple, text_position: tuple):
        self.zone_id = zone_id
        self.roi_points = np.array(roi_points, np.int32)
        self.config = config
        self.display_color = display_color
        self.text_position = text_position
        
        self.pixels_per_meter = self._calibrate()
        self.mask = None
        self.tracked_points = None
        self.frame_data = []
        self.current_speed_kph = 0
        self.good_new_points = []
        self.good_old_points = []

    def _calibrate(self) -> float:
        """Calculates the pixels-per-meter ratio for the zone."""
        distance_points = self.roi_points[:2]
        pixel_distance = np.linalg.norm(distance_points[0] - distance_points[1])
        ppm = pixel_distance / self.config.known_distance_meters
        print(f"Zone {self.zone_id} Calibration: {ppm:.2f} pixels per meter.")
        return ppm

    def initialize_features(self, gray_frame: np.ndarray):
        """Initializes features to track within the zone."""
        self.mask = np.zeros_like(gray_frame)
        cv2.fillPoly(self.mask, [self.roi_points], 255)
        self.tracked_points = cv2.goodFeaturesToTrack(gray_frame, mask=self.mask, **self.config.feature_params)

    def process_frame(self, old_gray: np.ndarray, current_gray: np.ndarray, frame_number: int, video_fps: float):
        """Processes a single frame to track points and calculate speed."""
        self.good_new_points, self.good_old_points = [], []
        
        if self.tracked_points is not None and len(self.tracked_points) > 0:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, current_gray, self.tracked_points, None, **self.config.lk_params)
            
            if new_points is not None:
                self.good_new_points = new_points[status == 1]
                self.good_old_points = self.tracked_points[status == 1]
                if len(self.good_new_points) > 0:
                    displacements = np.linalg.norm(self.good_new_points - self.good_old_points, axis=1)
                    avg_pixel_speed = np.mean(displacements)
                    speed_ms = (avg_pixel_speed / self.pixels_per_meter) * video_fps
                    self.current_speed_kph = speed_ms * 3.6
                else:
                    self.current_speed_kph = 0
                
                # Update points for the next frame
                self.tracked_points = self.good_new_points.reshape(-1, 1, 2)
            else:
                self.current_speed_kph = 0
                self.tracked_points = None # Lost all points
        else:
            self.current_speed_kph = 0

        # Log data for this frame
        self.frame_data.append({
            'frame_number': frame_number,
            'tracked_points_count': len(self.good_new_points),
            'speed_kph': round(self.current_speed_kph, 2)
        })

    def update_features(self, gray_frame: np.ndarray, force_update: bool = False):
        """Updates or re-initializes features."""
        if force_update or self.tracked_points is None or len(self.tracked_points) < 10:
             self.tracked_points = cv2.goodFeaturesToTrack(gray_frame, mask=self.mask, **self.config.feature_params)

    def draw_visuals(self, frame: np.ndarray):
        """Draws the zone, motion trails, and speed onto the frame."""
        # Draw semi-transparent zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi_points], self.display_color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        # Draw zone outline
        cv2.polylines(frame, [self.roi_points], isClosed=True, color=self.display_color, thickness=2)

        # Draw motion trails

        if len(self.good_new_points) > 0:
            for i, (new, old) in enumerate(zip(self.good_new_points, self.good_old_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw the line from old to new point
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
                # Draw a circle at the new position
                cv2.circle(frame, (int(a), int(b)), 1, (0, 255, 0), -1)
        # Draw speed text
        cv2.putText(frame, f"Zone {self.zone_id} Speed: {int(self.current_speed_kph)} km/h",
                    self.text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.display_color, 2, lineType=cv2.LINE_AA)

    def save_to_csv(self):
        """Saves the collected frame data to a CSV file."""
        output_path = f"{self.config.output_csv_prefix}_{self.zone_id}.csv"
        print(f"Writing data for Zone {self.zone_id} to '{output_path}'...")
        if not self.frame_data:
            print(f"No data to write for Zone {self.zone_id}.")
            return
            
        headers = ['frame_number', 'tracked_points_count', 'speed_kph']
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.frame_data)
            print(f"Zone {self.zone_id} CSV file successfully created.")
        except IOError as e:
            print(f"Error writing CSV for Zone {self.zone_id}: {e}")

def eight_pts_from_six(six_points):
    """Converts six points into two four-point polygons."""
    roi_points_1 = [six_points[2], six_points[3], six_points[5], six_points[4]]
    roi_points_2 = [six_points[2], six_points[3], six_points[1], six_points[0]]
    return roi_points_1, roi_points_2

def main():
    config = AppConfig()

    # --- Video and Zone Setup ---
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {config.video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    six_points = create_zones_from_video(config.video_path)
    if not six_points or len(six_points) != 6:
        print("Error: Zone creation failed. Using default zones.")
        six_points = [(289,466), (224,542), (439,466), (374,542), (589,466), (524,542)]
    
    roi_1_pts, roi_2_pts = eight_pts_from_six(six_points)

    # --- Initialize Zone Processors ---
    zones = [
        ZoneProcessor(zone_id=1, roi_points=roi_1_pts, config=config, display_color=(0, 200, 0), text_position=(frame_width - 300, 60)),
        ZoneProcessor(zone_id=2, roi_points=roi_2_pts, config=config, display_color=(200, 0, 0), text_position=(frame_width - 300, 90))
    ]

    # --- Initialize Video Writer and First Frame ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(config.output_video_path, fourcc, video_fps, (frame_width, frame_height))
    
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
        
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    for zone in zones:
        zone.initialize_features(old_gray)

    # --- Main Processing Loop ---
    frame_number = 0
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for zone in zones:
            # Process each zone
            zone.process_frame(old_gray, frame_gray, frame_number, video_fps)
            # Periodically re-detect features
            zone.update_features(frame_gray, force_update=(frame_number % 10 == 0))
            # Draw visualizations
            zone.draw_visuals(frame)

        video_writer.write(frame)
        old_gray = frame_gray.copy()

    # --- Cleanup and Save Data ---
    print("Releasing resources...")
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to '{config.output_video_path}'")

    for zone in zones:
        zone.save_to_csv()

if __name__ == "__main__":
    main()
