import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
plt.xticks([]), plt.yticks([])

def mul(A, B):
    h, w, _ = A.shape
    res = np.zeros((h, w, 2))
    res[:,:,0] = A[:,:,0] * B[:,:,0] - A[:,:,1] * B[:,:,1]
    res[:,:,1] = A[:,:,0] * B[:,:,1] + A[:,:,1] * B[:,:,0]
    return res

def square(A):
    h, w, _ = A.shape
    res = np.zeros((h, w))
    res = A[:,:,0] * A[:,:,0] + A[:,:,1] * A[:,:,1]
    return res

def shift(A):
    h, w = A.shape
    for x in range(h):
        for y in range(w):
            A[x, y] *= (-1) ** (x + y)

img1 = cv2.imread("./images/Photographer_degraded.tif", cv2.IMREAD_GRAYSCALE)
h, w = img1.shape
# shift(img1)
G = cv2.dft(np.float32(img1), flags = cv2.DFT_COMPLEX_OUTPUT)
T = 1
a = 0.1
b = 0.1
H = np.zeros((h, w, 2))
for x in range(h):
    for y in range(w):
        tmp = math.pi * (x * a + y * b)
        if x == 0 and y == 0:
            tmp = math.pi * (a + b)
        H[x, y, 0] = T * math.sin(tmp) * math.cos(-tmp) / tmp
        H[x, y, 1] = T * math.sin(tmp) * math.sin(-tmp) / tmp
F = np.zeros((h, w, 2))
F[:,:,0] = G[:,:,0] / H[:,:,0] - G[:,:,1] / H[:,:,1]
F[:,:,1] = G[:,:,0] / H[:,:,1] + G[:,:,1] / H[:,:,0]


res = cv2.idft(F)[:,:,0]
# shift(res)
plt.imshow(res, cmap = "gray")
plt.show()
