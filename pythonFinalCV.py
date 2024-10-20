import cv2
import sys
import time
import numpy as np
from collections import deque

# ------------------------------
# Initialize Video Capture
# ------------------------------
def initialize_camera(device_index=0):
    """
    Initialize the camera with the specified device index.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Could not open video capture device at index {device_index}.")
        sys.exit()
    return cap

# ------------------------------
# Initialize Tracking
# ------------------------------
def initialize_tracking():
    """
    Initialize data structures for tracking camera movement.
    """
    path = deque(maxlen=1000)    # To store the camera movement path
    return path

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Initialize camera
    cap = initialize_camera(device_index=0)  # Change to the correct device_index as needed

    # Initialize tracking data structures
    path = initialize_tracking()

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        sys.exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Adjusted Parameters for Shi-Tomasi Corner Detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,  # Reduced quality level for more features
                          minDistance=5,       # Reduced min distance between features
                          blockSize=7)

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # Retry mechanism if no features are found in the first frame
    while p0 is None:
        print("No features found in the first frame. Retrying...")
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to read frame during retry.")
            cap.release()
            sys.exit()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        time.sleep(0.5)  # Wait half a second before retrying

    # Initialize variables for motion tracking
    cumulative_translation = np.array([0, 0], dtype=np.float32)

    # Variables for inactivity detection
    last_move_time = time.time()
    inactive_positions = []  # List to store positions in scene coordinates (cumulative_translation)
    extended_inactivity_start_time = None  # Time when extended inactivity started

    # Set up OpenCV window
    window_name = "Microscope Live Feed"
    cv2.namedWindow(window_name)

    # Frame rate setup (20 FPS)
    fps = 20
    frame_duration = 1 / fps

    # Define the starting offset (higher on the screen)
    base_x = prev_frame.shape[1] // 2  # Center horizontally
    base_y = 100  # 100 pixels from the top; adjust as needed

    # Variable to keep track of whether the path should be drawn
    draw_path = True

    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow to track feature points
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0 and len(good_old) > 0:
                # Calculate the average movement
                translation = np.mean(good_new - good_old, axis=0)
                translation = -translation  # Invert the translation to match camera movement

                # Add a translation threshold to ignore minor jitters
                translation_magnitude = np.linalg.norm(translation)
                translation_threshold = 1.0  # Adjust this value as needed

                if translation_magnitude >= translation_threshold:
                    cumulative_translation += translation

                    # Append to path if drawing is enabled
                    if draw_path:
                        path.append(tuple(cumulative_translation))

                    # Update last_move_time
                    last_move_time = time.time()

                    # Reset extended inactivity timer if camera moves
                    extended_inactivity_start_time = None

                # Re-initialize feature points if necessary
                if len(good_new) < 500:
                    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
                    if p0 is not None:
                        good_new = p0
                        good_old = p0
            else:
                # If no good points are found, re-initialize
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        else:
            # If optical flow failed, re-initialize feature points
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

        # Update previous frame and points
        prev_gray = gray.copy()
        if p0 is not None and len(p0) > 0:
            p0 = p0.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

        # Draw the camera movement path if drawing is enabled
        if draw_path:
            for i in range(1, len(path)):
                if path[i - 1] is None or path[i] is None:
                    continue
                # Offset to set the starting point higher on the screen
                pt1 = (int(path[i - 1][0] + base_x), int(path[i - 1][1] + base_y))
                pt2 = (int(path[i][0] + base_x), int(path[i][1] + base_y))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Blue lines

        # Check for inactivity (no movement for >0.8 seconds)
        current_time = time.time()
        inactivity_time_threshold = 0.8  # Adjust as needed
        if (current_time - last_move_time) > inactivity_time_threshold:
            # Check if we need to start the extended inactivity timer
            if extended_inactivity_start_time is None:
                extended_inactivity_start_time = current_time

            # Store the inactivity position in scene coordinates (cumulative_translation) if not already stored
            if len(inactive_positions) == 0 or not np.array_equal(cumulative_translation, inactive_positions[-1]):
                inactive_position_scene = cumulative_translation.copy()
                inactive_positions.append(inactive_position_scene)
                print(f"Inactivity detected at scene position: {inactive_position_scene}")

            # Check for extended inactivity (no movement for >10 seconds)
            extended_inactivity_threshold = 10.0  # 10 seconds
            if (current_time - extended_inactivity_start_time) > extended_inactivity_threshold:
                if len(inactive_positions) >= 2:
                    # Delete the last two red dots
                    removed_positions = inactive_positions[-2:]
                    inactive_positions = inactive_positions[:-2]
                    print(f"Deleted last two red dots at positions: {removed_positions}")
                else:
                    # If fewer than two red dots, remove whatever is available
                    removed_positions = inactive_positions.copy()
                    inactive_positions.clear()
                    print(f"Deleted red dots at positions: {removed_positions}")
                # Reset extended inactivity timer
                extended_inactivity_start_time = None
        else:
            # Reset extended inactivity timer if camera moves
            extended_inactivity_start_time = None

        # Draw all stored red dots persistently, adjusted for current camera position
        for pos in inactive_positions:
            # Adjust the position based on current cumulative_translation
            delta_translation = pos - cumulative_translation
            screen_pos = (int(delta_translation[0] + base_x), int(delta_translation[1] + base_y))

            # Validate that the position is within frame boundaries before drawing
            if 0 <= screen_pos[0] < frame.shape[1] and 0 <= screen_pos[1] < frame.shape[0]:
                cv2.circle(frame, screen_pos, 5, (0, 0, 255), -1)  # Adjust radius as needed
            else:
                print(f"Red dot at {screen_pos} is out of frame bounds.")

        # Display the resulting frame
        cv2.imshow(window_name, frame)

        # Control Frame Rate
        elapsed_time = time.time() - start_time
        time_to_wait = frame_duration - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('r'):
            path.clear()
            cumulative_translation = np.array([0, 0], dtype=np.float32)
            inactive_positions.clear()
            extended_inactivity_start_time = None
            last_move_time = time.time()
            draw_path = True  # Reset draw_path
            print("Path and inactivity dots reset.")
        elif key == ord('e'):
            path.clear()
            draw_path = False  # Stop drawing the path
            print("Movement path cleared; red dots remain.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and destroyed all windows.")

if __name__ == "__main__":
    main()
