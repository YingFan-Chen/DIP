import cv2
import lib 
import sys
import numpy as np
img = cv2.imread("image_2.png", cv2.IMREAD_GRAYSCALE)
pz = np.full(256, 1 / 256)
histo = lib.HistogramMatch(img, pz)

h = np.size(img, 0)
w = np.size(img, 1)
ans = -1
dif = 256000000
for i in range(80, 100):
    g = i / 100
    gamma = lib.GammaTransformation(img, g)
    sum = int(0)
    for j in range(h):
        for k in range(w):
            sum += abs(gamma[i][j] - histo[i][j])
    if dif > sum :
        dif = sum
        ans = i
print(ans)