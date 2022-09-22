import lib
import cv2
import numpy as np

img = cv2.imread("image_2.png", cv2.IMREAD_GRAYSCALE)
res1 = lib.GammaTransformation(img, 0.67)
pz = np.full(256, 1 / 256)
res2 = lib.HistogramMatch(img, pz)

cv2.imshow("img", img)
cv2.imshow("res1", res1)
cv2.imshow("res2", res2)

cv2.waitKey(0)
