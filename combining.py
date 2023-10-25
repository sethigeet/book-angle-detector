from ultralytics import YOLO
import numpy as np
import cv2
import math


def auto_canny_edge_detection(img, sigma=0.33):
    md = np.median(img)
    lower = int(max(0, (1.0 - sigma) * md))
    upper = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(img, lower, upper)


model = YOLO("yolov8x.pt")

img = cv2.imread("./data/img1.png")
result = model.predict(img, conf=0.5)[0]
boxes = result.boxes
xyxys = boxes.xyxy
for xyxy in xyxys:
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])

    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    sub_img = img[y1:y2, x1:x2]

    # Edge detection
    gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny_edge_detection(blurred)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)

        x0n = a * r
        y0n = b * r

        x1n = int(x0n + 1000 * (-b))
        y1n = int(y0n + 1000 * (a))

        x2n = int(x0n - 1000 * (-b))
        y2n = int(y0n - 1000 * (a))

        cv2.line(sub_img, (x1n, y1n), (x2n, y2n), (0, 0, 255))
        if x2n == x1n:
            print("angle is 90 degrees")
        else:
            print(
                f"angle is {math.atan(-(y2n - y1n) / (x2n - x1n)) * 180 / math.pi} degrees"
            )

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
