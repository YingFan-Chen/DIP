import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
plt.xticks([]), plt.yticks([])

img1 = cv2.imread("./images/Photographer_degraded.tif", cv2.IMREAD_GRAYSCALE)
h, w = img1.shape
G = np.fft.fft2(img1)
G = np.fft.fftshift(G)
