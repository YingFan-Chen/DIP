import lib
import cv2
import numpy as np

img = cv2.imread("image_2.png", cv2.IMREAD_GRAYSCALE)
# gamma = 0.93 minimal
res1 = lib.GammaTransformation(img, 0.7)   
pz = np.full(256, 1 / 256)
res2 = lib.HistogramMatch(img, pz)

cv2.imwrite("./2img/res1.jpg", res1)
cv2.imwrite("./2img/res2.jpg", res2)
cv2.imshow("img", img)
cv2.imshow("res1", res1)
cv2.imshow("res2", res2)

cv2.waitKey(0)
