import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Define the line coordinates (can be modified as needed)
# Format: [x1, y1, x2, y2]
# Note: To move the line, modify these coordinates
counting_line = [50, 500, 600, 500]  # Initially at the bottom left part of the video

# Vehicle counter class to handle counting logic
class VehicleCounter:
    def __init__(self, counting_line):
        # Line format: [x1, y1, x2, y2]
        self.counting_line = counting_line
        self.count = 0
        self.already_counted = set()  # Store IDs of counted vehicles
        
    def update_line(self, new_line):
        """
        Update the counting line position
        Args:
            new_line: [x1, y1, x2, y2] - new line position
        """
        self.counting_line = new_line
        self.already_counted = set()  # Reset counted IDs when line changes
        self.count = 0
    
    def is_line_crossed(self, prev_centroid, curr_centroid):
        """
        Check if a centroid has crossed the line.
        Uses line segment intersection.
        
        Args:
            prev_centroid: (x, y) previous frame centroid
            curr_centroid: (x, y) current frame centroid
            
        Returns:
            True if line is crossed, False otherwise
        """
        # Line defined by counting_line [x1, y1, x2, y2]
        x1, y1, x2, y2 = self.counting_line
        
        # Path segment from previous to current position
        px1, py1 = prev_centroid
        px2, py2 = curr_centroid
        
        # Check line intersection
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Determine if two line segments intersect
        A = (x1, y1)
        B = (x2, y2)
        C = (px1, py1)
        D = (px2, py2)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def process_tracks(self, frame, tracks, track_history):
        """
        Process tracks and count vehicles crossing the line
        
        Args:
            frame: The current video frame
            tracks: Current frame tracks
            track_history: Dict storing previous positions, format {track_id: (x, y)}
            
        Returns:
            Annotated frame and updated track_history
        """
        # Make a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        # Draw the counting line
        x1, y1, x2, y2 = self.counting_line
        cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # (left, top, right, bottom)
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Calculate center point of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_centroid = (center_x, center_y)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(
                annotated_frame, 
                f"ID: {track_id}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            
            # Check if we have a previous position for this track
            if track_id in track_history:
                prev_centroid = track_history[track_id]
                
                # Check if the vehicle has crossed the line and hasn't been counted yet
                if track_id not in self.already_counted and self.is_line_crossed(prev_centroid, current_centroid):
                    self.count += 1
                    self.already_counted.add(track_id)
                    print(f"Vehicle ID {track_id} counted. Total count: {self.count}")
            
            # Update the history with current position
            track_history[track_id] = current_centroid
        
        # Add the counter information to the frame
        cv2.putText(
            annotated_frame, 
            f"Count: {self.count}", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        return annotated_frame, track_history

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None
    )
    
    # Open the video file
    video_path = "car_counting/video_raw/highway.mp4"  # Change this to your video file path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "car_counting/Output_videos/output_video_ant.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize the vehicle counter
    counter = VehicleCounter(counting_line)
    
    # Dictionary to store track history {track_id: (center_x, center_y)}
    track_history = {}
    
    print(f"Processing video: {video_path}")
    print(f"Press 'q' to quit, 'l' to change the counting line position")
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Run YOLOv8 detection
        results = model(frame, conf=0.25)[0]
        
        # Extract detections for vehicle classes (car=2, motorcycle=3, bus=5, truck=7)
        vehicle_classes = [2, 3, 5, 7]
        detections_for_tracker = []
        
        for data in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            if int(class_id) in vehicle_classes:
                # DeepSORT expects detections in format: [left, top, width, height]
                left, top = int(x1), int(y1)
                width, height = int(x2 - x1), int(y2 - y1)
                
                # DeepSORT expects a tuple with (bbox, confidence, feature)
                # where bbox is [left, top, width, height]
                bbox = [left, top, width, height]
                detection = (bbox, float(confidence), None)  # None for feature as it's computed internally
                detections_for_tracker.append(detection)
        
        # Update tracker
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        
        # Process tracks for counting
        if tracks:
            frame, track_history = counter.process_tracks(frame, tracks, track_history)
        
        # Display the frame
        cv2.imshow("Vehicle Counting", frame)
        out.write(frame)
        
        # Press 'q' to exit, 'l' to modify the counting line
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('l'):
            # Example: Change line position when 'l' is pressed
            new_line = [100, 400, 700, 400]  # [x1, y1, x2, y2]
            counter.update_line(new_line)
            print("Line position updated!")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()