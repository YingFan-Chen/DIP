import lib
import cv2
import numpy as np

img = cv2.imread("image_4.tif", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(img.shape)
tmp1 = lib.Resize(img, h / 4, w / 4, 2)
res1 = lib.GaussainFiliter(tmp1, 128, 128, 1)
res2 = lib.Divide(tmp1, res1)
cv2.imwrite("A.jpg", res2)
res2 = lib.GammaTransformation(res2, 0.5)
# res1 = lib.Resize(res1, h, w, 2)
# res2 = lib.Resize(res2, h, w, 2)
cv2.imshow("res1", res1)
cv2.imshow("res2", res2)
cv2.waitKey(0)

img = cv2.imread("A.jpg", cv2.IMREAD_GRAYSCALE)
cmp = cv2.imread("image_4.tif", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
img = lib.GammaTransformation(img, 0.017)
img = lib.Resize(img, h * 4, w * 4, 0)
cv2.imshow("res", img)
cv2.imshow("cmp", cmp)
cv2.waitKey(0)
