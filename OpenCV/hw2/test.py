import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
plt.xticks([]), plt.yticks([])

def H(a, b, size, T):
    res = np.zeros(size, dtype="complex")
    M, N = size
    for u in range(M):
        for v in range(N):
            Q = math.pi * (u * a + v * b)
            res[u, v] = math.sin(Q) * math.e ** (complex(0, -Q)) + complex(1e-31, 1e-31)
    return res

def motion_blur(size, angle):
    res= np.zeros((size, size))
    res[(size - 1)//2 ,:] = np.ones(size)
    res = cv2.warpAffine(res, cv2.getRotationMatrix2D((size/2 - 0.5 , size/2 - 0.5) , angle, 1.0), (size, size))  
    res = res / np.sum(res)       
    return res

size = (256,256)
H = H(10, 10, size)
plt.subplot(121)
plt.imshow(20*np.log(abs(H)), cmap = "gray")
m = motion_blur(10, 315)
M = np.fft.fft2(m, s = size)
plt.subplot(122)
plt.imshow(20*np.log(abs(M)), cmap = "gray")
plt.show()