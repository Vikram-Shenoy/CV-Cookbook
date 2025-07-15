import cv2
import numpy as np
import sys
import os
import argparse

# --- Configuration ---
# You can adjust these values
POINT_OFFSET = 20  
ZONE_OFFSET_X = 150  # The horizontal distance to create the parallel lines
LINE_THICKNESS = 2
CIRCLE_RADIUS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors (B, G, R format)
POINT_COLOR = (0, 255, 255)  # Yellow for clicked points
LINE_COLOR = (0, 0, 255)     # Red for zone boundaries
ZONE_1_COLOR = (255, 0, 0)   # Blue for the left zone
ZONE_2_COLOR = (0, 255, 0)   # Green for the right zone
TRANSPARENCY_ALPHA = 0.3     # Transparency of the shaded zones

# Global list to store the user-clicked points
points = []
frame_clone = None

def display_instructions(image, num_points):
    """Displays user instructions on the image."""
    h, w, _ = image.shape
    
    if num_points == 0:
        text = "Click to select the FIRST point for the central line."
    elif num_points == 1:
        text = "Click to select the SECOND point for the central line."
    else: # num_points == 2
        text = "Points selected. Press 'C' to confirm, 'R' to reset, or 'ESC' to exit."
        
    # Add a semi-transparent background for the text
    text_size, _ = cv2.getTextSize(text, FONT, 0.7, 2)
    text_w, text_h = text_size
    cv2.rectangle(image, (10, 10), (10 + text_w + 10, 10 + text_h + 10), (0,0,0), -1)
    cv2.putText(image, text, (15, 15 + text_h), FONT, 0.7, (255, 255, 255), 2)


def select_points_callback(event, x, y, flags, param):
    """Mouse callback function to capture points."""
    global frame_clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x-20, y))
            # Draw a circle on the clone to give visual feedback
            cv2.circle(frame_clone, (x, y), CIRCLE_RADIUS, POINT_COLOR, -1)
            print(f"Point {len(points)} selected: ({x}, {y})")


def create_zones_from_video(video_path):
    """
    Main function to handle video loading, user interaction, and zone creation.
    """
    global frame_clone, points

    # 1. Read the first frame of the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        cap.release()
        sys.exit(1)
        
    cap.release()
    
    # Create a clone of the frame to draw on, so we can reset it
    original_frame = frame.copy()
    frame_clone = original_frame.copy()

    # 2. Setup window and mouse callback
    window_name = "Zone Creator - Click two points"
    cv2.namedWindow(window_name)

    cv2.setMouseCallback(window_name, select_points_callback)

    print("--- Zone Creation Utility ---")
    print("Please select two points on the frame to define the central line.")
    print("Press 'C' to confirm selection.")
    print("Press 'R' to reset points.")
    print("Press 'ESC' to exit.")

    # 3. Interaction loop
    while True:
        # Display instructions on the current frame
        temp_display_frame = frame_clone.copy()
        display_instructions(temp_display_frame, len(points))
        cv2.imshow(window_name, temp_display_frame)

        key = cv2.waitKey(20) & 0xFF

        # ESC key to exit
        if key == 27:
            print("\nExiting program gracefully.")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return None

        # 'r' key to reset
        elif key == ord('r'):
            print("\nResetting points. Please start over.")
            points = []
            frame_clone = original_frame.copy()

        # 'c' key to confirm
        elif key == ord('c'):
            if len(points) == 2:
                print("\nTwo points selected. Proceeding to create zones.")
                break
            else:
                print("\nError: You must select exactly two points before pressing 'C'. Exiting Zone Creator.")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return None
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Wait for a brief moment to ensure the window closes
    # 4. Calculate the 6 points for the two parallelogram zones
    p1, p2 = points[0], points[1]

    # The 6 points are based on a horizontal offset from the central line
    # Left Zone points
    p_left_1 = (p1[0] - ZONE_OFFSET_X, p1[1])
    p_left_2 = (p2[0] - ZONE_OFFSET_X, p2[1])

    # Right Zone points
    p_right_1 = (p1[0] + ZONE_OFFSET_X, p1[1])
    p_right_2 = (p2[0] + ZONE_OFFSET_X, p2[1])
    
    # The final 6 points to be returned.
    # [Top-Left, Bottom-Left, Top-Center, Bottom-Center, Top-Right, Bottom-Right]
    final_six_points = [p_left_1, p_left_2, p1, p2, p_right_1, p_right_2]
    
    # Print the points to the console (for the calling script)
    print("\n--- Calculated Zone Points ---")
    # Convert points to a string format for printing
    points_str = ', '.join([f'({pt[0]},{pt[1]})' for pt in final_six_points])
    print(points_str)
    print("----------------------------\n")

    # 5. Create and display the output image with shaded zones
    output_image = original_frame.copy()
    overlay = output_image.copy()

    # Define the polygons for shading
    zone1_poly = np.array([p_left_1, p_left_2, p2, p1], np.int32)
    zone2_poly = np.array([p1, p2, p_right_2, p_right_1], np.int32)

    # Draw the filled, transparent polygons on the overlay
    cv2.fillPoly(overlay, [zone1_poly], ZONE_1_COLOR)
    cv2.fillPoly(overlay, [zone2_poly], ZONE_2_COLOR)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, TRANSPARENCY_ALPHA, output_image, 1 - TRANSPARENCY_ALPHA, 0, output_image)
    
    # Draw the boundary lines on top of the blended image
    # Left line
    cv2.line(output_image, p_left_1, p_left_2, LINE_COLOR, LINE_THICKNESS)
    # Center line
    cv2.line(output_image, p1, p2, LINE_COLOR, LINE_THICKNESS)
    # Right line
    cv2.line(output_image, p_right_1, p_right_2, LINE_COLOR, LINE_THICKNESS)
    # Connecting top and bottom lines
    cv2.line(output_image, p_left_1, p_right_1, LINE_COLOR, LINE_THICKNESS)
    cv2.line(output_image, p_left_2, p_right_2, LINE_COLOR, LINE_THICKNESS)

    # Define the output directory
    output_dir = "2. vehicle_speed_detection/frames/output"
    
    # Create the directory and any necessary parent directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path for the output image
    output_filename = "zones_example.png"
    full_output_path = os.path.join(output_dir, output_filename)

    # Save the output image to the specified path
    cv2.imwrite(full_output_path, output_image)
    print(f"Example zone image saved as '{full_output_path}'")
    
    # Display the final image
    cv2.imshow("Final Zones Created", output_image)
    print("Press any key to close the final image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Wait for a brief moment to ensure the window closes
    
    return final_six_points


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Create two adjacent parallelogram zones based on two user-selected points on a video's first frame."
    # )
    # parser.add_argument("video_path", type=str, help="Path to the input mp4 video file.")
    # args = parser.parse_args()
    video_path = "2. vehicle_speed_detection/videos/input/highway_clipped.mp4"
    points = create_zones_from_video(video_path)
    print(points)
    # return points