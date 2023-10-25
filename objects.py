from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8x-seg.pt")

img = cv2.imread("./data/img1.png")
print(img.shape)
results = model.predict(img, conf=0.5)
result = results[0]
masks = result.masks
xys = masks.xy[0]
print(np.where(xys > 0.5, 1, 0).shape)

# segmentation_mask = masks[0]
# # Convert the segmentation mask to a binary mask
# binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
# print(binary_mask[..., np.newaxis])
# white_background = np.full(img.shape, 255)

# # Apply the binary mask
# new_image = (
#     white_background * (1 - binary_mask[..., np.newaxis])
#     + img * binary_mask[..., np.newaxis]
# )

# cv2.imshow("new image", new_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
