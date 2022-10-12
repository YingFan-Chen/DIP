import cv2
import numpy as np
import lib
import math

T = cv2.imread("T.png", cv2.IMREAD_GRAYSCALE)
T_transformed = cv2.imread("T_transformed.png", cv2.IMREAD_GRAYSCALE)
h, w = T.shape
res = lib.Resize(T, math.floor(h * 0.77), math.floor(w * 0.77), lib.Cubic)
res = lib.Rotate(res, h, w, 15, lib.Cubic)

cv2.imwrite("./3img/res.jpg", res)
cv2.imshow("res", res)
cv2.imshow("T_transformed", T_transformed)
cv2.waitKey(0)