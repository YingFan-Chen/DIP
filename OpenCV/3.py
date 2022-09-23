import cv2
import numpy as np
import lib

img = cv2.imread("T.png", cv2.IMREAD_GRAYSCALE)
h = np.size(img, 0)
w = np.size(img, 1)
res = lib.Resize(img, h << 1, w << 1, "Nearest_Neighbor")

cv2.imshow("img", img)
cv2.imshow("res", res)
cv2.waitKey(0)