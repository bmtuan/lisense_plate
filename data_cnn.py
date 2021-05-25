import imutils
from skimage.filters import threshold_local
from CCA import auto_canny
from yolo import get_box_coord
import numpy as np
import cv2
from imutils import perspective
import os

list_img = os.listdir('0')
print(list_img)
index = 0
for dx in range(0, 400):
    image = cv2.imread('0/' + list_img[dx])
    result = get_box_coord(image)
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = result[0]

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    LpRegion = perspective.four_point_transform(image, rect)
    V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 25, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh, 5)

    edges = auto_canny(thresh)

    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = thresh.shape[0] * thresh.shape[1]
    print(list_img[dx])
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        roi_ratio = roi_area / img_area
        if 0.01 <= roi_ratio <= 0.09 and 110 <= h <= 150:
            tmp = thresh[y - 3:y + h + 3, x - 3:x + w + 3]
            cv2.imwrite('data0/' + str(index) + '.jpg', tmp)
            print(str(index) + '.jpg')
            index = index + 1
