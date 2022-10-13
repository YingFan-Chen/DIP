import cv2
from matplotlib.ft2font import KERNING_UNSCALED
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
plt.xticks([]), plt.yticks([])
h, w = img.shape
print(img.shape)

# (a)
old = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
old = np.fft.fftshift(old)
plt.imshow(20*np.log(cv2.magnitude(old[:,:,0], old[:,:,1])), cmap = "gray")
plt.show()

for i in range(h):
    for j in range(w):
        img[i, j] *= (-1) ** (i + j)
new = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
'''for i in range(h):
    for j in range(w):
        new[i, j] *= (-1) ** (i + j)'''
plt.imshow(20*np.log(cv2.magnitude(new[:,:,0], new[:,:,1])), cmap = "gray")
plt.show()

# (b)
kernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
odd_symmetry = np.zeros((4, 4))
odd_symmetry[1:4,1:4] = kernal
padding = np.zeros((h, w))
padding[h//2-2:h//2+2, w//2-2:w//2+2] = odd_symmetry
old_padding = padding

for i in range(h):
    for j in range(w):
        padding[i, j] *= (-1) ** (i + j)

dft1 = cv2.dft(np.float32(padding), flags = cv2.DFT_COMPLEX_OUTPUT)

for i in range(h):
    for j in range(w):
        dft1[i, j, 1] *= (-1) ** (i + j)

plt.imshow(dft1[:,:,1], cmap = "gray")
plt.show()

res_dft = np.zeros((h, w, 2))
res_dft[:,:,0] = - old[:,:,1] * dft1[:,:,1]
res_dft[:,:,1] = old[:,:,0] * dft1[:,:,1]
res_dft = np.fft.ifftshift(res_dft)
res1 = cv2.idft(res_dft)
plt.imshow(res1[:,:,0], cmap = "gray")
plt.show()

res_dft = np.zeros((h, w, 2))
res_dft[:,:,0] = - new[:,:,1] * dft1[:,:,1]
res_dft[:,:,1] = new[:,:,0] * dft1[:,:,1]
res2 = cv2.idft(res_dft)[:,:,0]
for i in range(h):
    for j in range(w):
        res2[i, j] *= (-1) ** (i + j)
plt.imshow(res2, cmap = "gray")
plt.show()