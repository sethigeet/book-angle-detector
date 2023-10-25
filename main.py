from ultralytics import YOLO
import numpy as np
import cv2
import math

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)


def auto_canny_edge_detection(img, sigma=0.33):
    md = np.median(img)
    lower = int(max(0, (1.0 - sigma) * md))
    upper = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(img, lower, upper)


model = YOLO("yolov8x-seg.pt")

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
    if lines is None:
        continue

    angles = []
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        angles.append(theta)

    avg_angle = sum(angles) / len(angles)
    # convert to degrees
    avg_angle = int(avg_angle * 180 / math.pi)

    cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 1)

    (w, h), _ = cv2.getTextSize(str(avg_angle), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), BLUE, -1)
    img = cv2.putText(
        img, str(avg_angle), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1
    )

cv2.imshow("Detection", img)
cv2.imwrite("./output.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
