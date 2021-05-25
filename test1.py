import imutils
from skimage.filters import threshold_local
from CCA import auto_canny
from yolo import get_box_coord
import numpy as np
import cv2
from imutils import perspective
import keras


def predict(image):
    result = get_box_coord(image)
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = result[0]

    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])
    box = image[round(y_min):round((y_min+height)),round(x_min):round(x_min+width)]
    LpRegion = perspective.four_point_transform(image, rect)
    V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 25, offset=10, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh, 5)
    edges = auto_canny(thresh)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = thresh.shape[0] * thresh.shape[1]
    list = []
    coords = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        roi_ratio = roi_area / img_area
        if 0.01 <= roi_ratio <= 0.09 and 110 <= h <= 150:
            tmp = thresh[y - 3:y + h + 3, x - 3:x + w + 3]
            tmp = cv2.resize(tmp, (32, 64), interpolation=cv2.INTER_AREA)
            list.append(tmp)
            coords.append((y, x))
    first_line = []
    second_line = []
    candidates = []
    model = keras.models.load_model('my_model_12052021')
    ch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S", "T",
          "U",
          "V",
          "X", "Y", "Z"]
    for i in range(0, len(list)):
        arr = model.predict(list[i].reshape(1, 64, 32, 1)).astype('float32')
        char = str(ch[np.argmax(arr)])
        candidates.append(char)
    def take_first(s):
        return s[0]
    if (len(coords)):
        coords_min = min(coords, key=take_first)
        for i in range(len(list)):
            if coords_min[0] + 40 > coords[i][0]:
                first_line.append((candidates[i], coords[i][1]))
            else:
                second_line.append((candidates[i], coords[i][1]))
        text = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])
    else:
        text = ''
    pos = (int(rect[0][0]) - 50, int(rect[0][1]))
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return [box,thresh, image]

