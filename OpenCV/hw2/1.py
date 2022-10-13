import cv2
from matplotlib.ft2font import KERNING_UNSCALED
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
plt.xticks([]), plt.yticks([])
h, w = img.shape
print(img.shape)

# (a)
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft = np.fft.fftshift(dft)
dft_magnitude = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))
'''
plt.imshow(dft_magnitude, cmap = "gray")
plt.title("Fourier Spectrum of Keyboard")
plt.show()
'''
plt.imsave("./images/1_a.jpg", dft_magnitude, cmap = "gray")

# (b)
kernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
odd_symmetry = np.zeros((4, 4))
odd_symmetry[1:4,1:4] = kernal
padding = np.zeros((h, w))
padding[h//2-2:h//2+2, w//2-2:w//2+2] = odd_symmetry
for i in range(h):
    for j in range(w):
        padding[i, j] *= (-1) ** (i + j)

filter_dft = cv2.dft(np.float32(padding), flags = cv2.DFT_COMPLEX_OUTPUT)
for i in range(h):
    for j in range(w):
        filter_dft[i, j, 1] *= (-1) ** (i + j)
        filter_dft[i, j, 0] = 0
'''
plt.imshow(filter_dft[:,:,1], cmap = "gray")
plt.title("Frequency Domain Filter Transfer Function")
plt.show()
'''
plt.imsave("./images/1_b.jpg", filter_dft[:,:,1], cmap = "gray")

# (c)
res_dft = np.zeros((h, w, 2))
res_dft[:,:,0] = - dft[:,:,1] * filter_dft[:,:,1]
res_dft[:,:,1] = dft[:,:,0] * filter_dft[:,:,1]
res_dft = np.fft.ifftshift(res_dft)
res = cv2.idft(res_dft)
res_magnitude = cv2.magnitude(res[:,:,0], res[:,:,1])
'''
plt.imshow(res[:,:,0], cmap = "gray")
plt.title("Result After Frequency Domain Filtering")
plt.show()
'''
plt.imsave("./images/1_c.jpg", res[:,:,0], cmap = "gray")

# (d)
padding = np.zeros((h + 4, w + 4))
padding[2:h+2,2:w+2] = img
reverse_kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
new_res = np.zeros((h, w))
for x in range(h):
    for y in range(w):
        for i in range(3):
            for j in range(3):
                new_res[x, y] += padding[x + i, y + j] * reverse_kernal[i, j]
'''
plt.imshow(new_res, cmap = "gray")
plt.title("Result After Spatial Domain Filtering")
plt.show()
'''
plt.imsave("./images/1_d.jpg", new_res, cmap = "gray")

# (e)
kernal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
odd_symmetry = np.zeros((4, 4))
odd_symmetry[0:3,0:3] = kernal
padding = np.zeros((h, w))
padding[h//2-2:h//2+2, w//2-2:w//2+2] = odd_symmetry
for i in range(h):
    for j in range(w):
        padding[i, j] *= (-1) ** (i + j)

filter_dft = cv2.dft(np.float32(padding), flags = cv2.DFT_COMPLEX_OUTPUT)
for i in range(h):
    for j in range(w):
        filter_dft[i, j, 1] *= (-1) ** (i + j)
        filter_dft[i, j, 0] = 0

res_dft = np.zeros((h, w, 2))
res_dft[:,:,0] = - dft[:,:,1] * filter_dft[:,:,1]
res_dft[:,:,1] += dft[:,:,0] * filter_dft[:,:,1]
res_dft = np.fft.ifftshift(res_dft)
res = cv2.idft(res_dft)
res_magnitude = cv2.magnitude(res[:,:,0], res[:,:,1])
'''
plt.imshow(res[:,:,0], cmap = "gray")
plt.title("Result After Frequency Domain Filtering")
plt.show()
'''
plt.imsave("./images/1_e.jpg", res[:,:,0], cmap = "gray")