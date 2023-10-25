import cv2
import numpy as np


def auto_canny_edge_detection(img, sigma=0.33):
    md = np.median(img)
    lower = int(max(0, (1.0 - sigma) * md))
    upper = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(img, lower, upper)


# Create a VideoCapture object to capture video from the default camera (usually 0)
frame = cv2.imread("./data/img1.png")

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = auto_canny_edge_detection(blurred)

# This returns an array of r and theta values
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# The below for loop runs till r and theta values
# are in the range of the 2d array
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * r
    y0 = b * r

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Edge detection", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
