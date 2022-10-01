import lib
import cv2
import numpy as np
import math

img = cv2.imread("image_4.tif", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

sigma = 128 * (2 ** (1 / 2))
size = 256

if size % 2 == 0:
    size += 1

mid = math.floor(size / 2)

column = np.zeros(size)
sum = 0
for i in range(size):
    column[i] = math.e ** (-((i - mid) ** 2) / (2 * sigma ** 2))
    sum += column[i]
for i in range(size):
    column[i] /= sum

row = np.zeros(size)
sum = 0
for i in range(size):
    row[i] = math.e ** (-((i - mid) ** 2) / (2 * sigma ** 2))
    sum += row[i]
for i in range(size):
    row[i] /= sum

pad = lib.Padding(img, mid)
res = np.zeros((h, w), dtype = "uint8")
for i in range(h):
    for j in range(w):
        sum = 0
        for k in range(size):
            sum += pad[mid + i, j + k] * row[k]
        res[i, j] = sum

pad = lib.Padding(res, mid)
for i in range(h):
    for j in range(w):
        sum = 0
        for k in range(size):
            sum += pad[i + k, mid + j] * column[k]
        res[i, j] = sum
res1 = lib.Divide(img, res)
res1 = lib.GammaTransformation(res1, 0.05)
res = lib.GammaTransformation(res, 0.3)
cv2.imshow("res", res)
cv2.imshow("res1", res1)

cv2.waitKey(0)