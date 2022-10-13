from argparse import Namespace
from tkinter import W
import cv2
from matplotlib.ft2font import KERNING_UNSCALED
import numpy as np
import matplotlib.pyplot as plt
import math
# Function
def ILPT(P, Q, D):
    res = np.zeros((P, Q))
    mid = [P/2, Q/2]
    for i in range(P):
        for j in range(Q):
            dis = math.dist(mid, [i, j])
            if dis <= D:
                res[i, j] = 1
    return res

def GLPT(P, Q, D):
    res = np.zeros((P, Q))
    mid = [P/2, Q/2]
    for i in range(P):
        for j in range(Q):
            dis = math.dist(mid, [i, j])
            res[i, j] = math.e ** (- dis ** 2 / (2 * D ** 2))
    return res

def BLPT(P, Q, D, n = 1):
    res = np.zeros((P, Q))
    mid = [P/2, Q/2]
    for i in range(P):
        for j in range(Q):
            dis = math.dist(mid, [i, j])
            res[i, j] = 1 / (1 + (dis / D) ** (2 * n))
    return res

def Laplacian(P, Q):
    res = np.zeros((P, Q))
    mid = [P/2, Q/2]
    for i in range(P):
        for j in range(Q):
            dis = math.dist(mid, [i, j])
            res[i, j] = - 4 * (math.pi ** 2) * (dis ** 2) 
    return res

def Spec1(P, Q):
    res = np.ones((P, Q))
    for i in range(Q):
        res[P // 2, i] = 0.1
    return res

def LPF(img1, filter = "ILPT", ratio = 1, n = 1):
    h1, w1 = img1.shape
    P = h1 * 2
    Q = w1 * 2 
    padding = np.zeros((P, Q))
    padding[h1//2:h1//2+h1, w1//2:w1//2+w1] = img1
    for i in range(P):
        for j in range(Q):
            padding[i, j] *= (-1) ** (i + j)

    dft = cv2.dft(np.float32(padding), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_magnitude = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))

    if filter == "GLPT":
        dft_filter = GLPT(P, Q, min(P, Q) * ratio)
    elif filter == "BLPT":
        dft_filter = BLPT(P, Q, min(P, Q) * ratio, n)
    elif filter == "Laplacian":
        dft_filter = Laplacian(P, Q)
    elif filter == "Spec1":
        dft_filter = Spec1(P, Q)
    else:
        dft_filter = ILPT(P, Q, min(P, Q) * ratio)

    dft_res = np.zeros((P, Q, 2))
    dft_res[:,:,0] = dft[:,:,0] * dft_filter
    dft_res[:,:,1] = dft[:,:,1] * dft_filter
    res_magnitude = 20*np.log(cv2.magnitude(dft_res[:,:,0], dft_res[:,:,1]))
    plt.imshow(res_magnitude, cmap = "gray")
    plt.show()

    res = cv2.idft(dft_res)[:,:,0]
    for i in range(P):
        for j in range(Q):
            res[i, j] *= (-1) ** (i + j)
    res = res[h1//2:h1//2+h1, w1//2:w1//2+w1]
    return res

def HPF(img1, filter = "ILPT", ratio = 1, n = 1):
    h1, w1 = img1.shape
    P = h1 * 2
    Q = w1 * 2 
    padding = np.zeros((P, Q))
    padding[h1//2:h1//2+h1, w1//2:w1//2+w1] = img1
    for i in range(P):
        for j in range(Q):
            padding[i, j] *= (-1) ** (i + j)

    dft = cv2.dft(np.float32(padding), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_magnitude = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))
    plt.imshow(dft_magnitude, cmap = "gray")
    plt.show()
    if filter == "GLPT":
        dft_filter = 1 - GLPT(P, Q, min(P, Q) * ratio)
    elif filter == "BLPT":
        dft_filter = 1 - BLPT(P, Q, min(P, Q) * ratio, n)
    elif filter == "Laplacian":
        dft_filter = 1 - Laplacian(P, Q)
    else:
        dft_filter = 1 - ILPT(P, Q, min(P, Q) * ratio)

    dft_res = np.zeros((P, Q, 2))
    dft_res[:,:,0] = dft[:,:,0] * dft_filter
    dft_res[:,:,1] = dft[:,:,1] * dft_filter

    res = cv2.idft(dft_res)[:,:,0]
    for i in range(P):
        for j in range(Q):
            res[i, j] *= (-1) ** (i + j)
    res = res[h1//2:h1//2+h1, w1//2:w1//2+w1]
    return res

# Main
img1 = cv2.imread("./images/Einstein.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./images/phobos.tif", cv2.IMREAD_GRAYSCALE)
plt.xticks([]), plt.yticks([])
print(img1.shape)
print(img2.shape)

# Einstein
'''
res = HPF(img1, "BLPT", 0.0001)
plt.imshow(res, cmap = "gray")
plt.show()
'''

# phobos
res = HPF(img2, "ILPT", 0.005)
plt.imshow(res, cmap = "gray")
plt.show()