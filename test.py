import cv2
import numpy as np

# Initialize variables to store selected points
points = []

# Mouse callback function to capture points
def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked point
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        # If two points are selected, print the distance
        if len(points) == 2:
            distance = calculate_distance(points[0], points[1])
            print(f"Distance between points: {distance} pixels")
            calculate_xy(points[0], points[1])

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return int(np.linalg.norm(np.array(point1) - np.array(point2)))

def calculate_xy(point1, point2):
    result = []
    for i in range(len(point1)):
        result.append(point1[i]-point2[i])
    print(result)


# Load the grid image
image_path = 'Screenshot 2024-10-19 153209.png'  # Replace with your grid image path
image = cv2.imread(image_path)

# Set up the mouse callback
cv2.namedWindow('Grid Image')
cv2.setMouseCallback('Grid Image', select_point)

while True:
    # Display the image
    cv2.imshow('Grid Image', image)

    # Check for a key press
    key = cv2.waitKey(1)
    if key == 27:  # Esc key to exit
        break

# Clean up
cv2.destroyAllWindows()