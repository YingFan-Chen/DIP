import cv2
import math
from cv2 import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
plt.xticks([]), plt.yticks([])

def GLPT(P, Q, D):
    res = np.zeros((P, Q))
    mid = [P/2, Q/2]
    for i in range(P):
        for j in range(Q):
            dis = math.dist(mid, [i, j])
            res[i, j] = math.e ** (- dis ** 2 / (2 * D ** 2))
    return res

def motion_blur(size, angle):
    res= np.zeros((size, size))
    res[(size - 1)//2 ,:] = np.ones(size)
    res = cv2.warpAffine(res, cv2.getRotationMatrix2D((size/2 - 0.5, size/2 - 0.5) , angle, 1.0), (size, size))  
    res = res / np.sum(res)       
    return res

def wiener_filter(img, kernal, k):
    kernal = kernal / np.sum(kernal)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernal = np.fft.fft2(kernal, s = img.shape)
    kernal = np.conj(kernal) / (np.abs(kernal) ** 2 + k)
    dummy = dummy * kernal
    dummy = np.fft.ifft2(dummy)
    return dummy.real

img1 = cv2.imread("./images/Photographer_degraded.tif", cv2.IMREAD_GRAYSCALE)
h, w = img1.shape
print(img1.shape)
kernal = motion_blur(5, 45)
NSR = 0.001
tmp = wiener_filter(img1, kernal, NSR)
tmp = np.fft.fft2(tmp)
tmp = np.fft.fftshift(tmp)
G = GLPT(h, w, 10)
tmp = tmp * G
tmp = np.fft.fftshift(tmp)
tmp = np.fft.ifft2(tmp).real
plt.imsave("./images/4_Photographer_restored.jpg", tmp, cmap = "gray", vmin=0, vmax=255)

img2 = cv2.imread("./images/Football players_degraded.tif", cv2.IMREAD_GRAYSCALE)
h, w = img2.shape
print(img2.shape)
kernal = motion_blur(10, 40)
NSR = 0.001
tmp = wiener_filter(img2, kernal, NSR)
plt.imsave("./images/4_Football players_restored.jpg", tmp, cmap = "gray", vmin=0, vmax=255)