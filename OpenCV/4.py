import lib
import cv2
import numpy as np

img = cv2.imread("image_4.tif", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

tmp1 = lib.Resize(img, h / 4, w / 4, 2)
res1 = lib.GaussainFiliter(tmp1, 32, 1024, 1)
res2 = lib.Divide(tmp1, res1)
res2 = lib.GammaTransformation(res2, 0.017)
res1 = lib.Resize(res1, h, w, 0)
res2 = lib.Resize(res2, h, w, 0)
cv2.imwrite("./4img/32_sigma1024_1.jpg", res1)
cv2.imwrite("./4img/32_sigma1024_2.jpg", res2)
cv2.imshow("res1", res1)
cv2.imshow("res2", res2)
cv2.waitKey(0)