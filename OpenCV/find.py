# This script is for finding the Gamma value which makes minimal difference between Histogram Matching and Gamma Transformation
# Gamma value 0 - 1 (interval 0.01)
import cv2
import lib 
import numpy as np
img = cv2.imread("image_2.png", cv2.IMREAD_GRAYSCALE)
pz = np.full(256, 1 / 256)
histo = lib.HistogramMatch(img, pz)

h, w = img.shape
res = -1
dif = 256000000
for i in range(0, 100):
    g = i / 100
    gamma = lib.GammaTransformation(img, g)
    sum = int(0)
    for j in range(h):
        for k in range(w):
            sum += abs(gamma[i][j] - histo[i][j])
    if dif > sum :
        dif = sum
        res = i
print(res)