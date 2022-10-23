import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
plt.xticks([]), plt.yticks([])

# (u, v) : notch center, (P, Q) : dft.shape, (d, n) : Butterworth parameters
# lower pass filter
def Butterworth(u, v, d, n, h, w):
    def Dk(x, y, u, v):
        return ((x - u) ** 2 + (y - v) ** 2) ** 0.5
    res = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            res[x, y] = (1 / (1 + (Dk(x, y, u, v) / d) ** n)) * (1 / (1 + (Dk(h - x, w - y, u, v) / d) ** n))
    return res

def Ideal(u, v, d, h, w):
    def Dk(x, y, u, v):
        return ((x - u) ** 2 + (y - v) ** 2) ** 0.5
    res = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            if Dk(x, y, u, v) > d:
                res[x, y] = 0
            else:
                res[x, y] = 1
    return res

img = cv2.imread("./images/Martian terrain.tif", cv2.IMREAD_GRAYSCALE)
print(img.shape)
h, w = img.shape

shift = np.zeros((h, w))
for x in range(h):
    for y in range(w):
        shift[x, y] = img[x, y] # * (-1) ** (x + y)
dft = cv2.dft(np.float32(shift), flags = cv2.DFT_COMPLEX_OUTPUT)

# show spectrum image(weird that different from textbook)
mag = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))
for x in range(h):
    for y in range(w):
        if mag[x, y] < 180:
            mag[x, y] = 0
plt.imshow(mag, cmap = "gray")
plt.show()

# To Do
d = 10
n = 1
filter = np.ones((h, w))
filter *= Butterworth(h//2, 0, n, d, h, w)
filter *= Butterworth(h//2, w, n, d, h, w)
filter *= Butterworth(0, w//2, n, d, h, w)
filter *= Butterworth(h, w//2, n, d, h, w)

plt.imshow(filter, cmap = "gray")
plt.show()
            
N = np.zeros((h, w, 2))
N[:,:,0] = filter * dft[:,:,0]
N[:,:,1] = filter * dft[:,:,1]
plt.imshow(20*np.log(cv2.magnitude(N[:,:,0], N[:,:,1])), cmap = "gray")
plt.show()

eta = cv2.idft(N)[:,:,0]
for x in range(h):
    for y in range(w):
        eta[x, y] = eta[x, y] # * (-1) ** (x + y)

weight = np.zeros((h, w))
pad_img = np.zeros((h + 4, w + 4))
pad_img[2:h+2, 2:w+2] = img
pad_eta = np.zeros((h + 4, w + 4))
pad_eta[2:h+2, 2:w+2] = eta
for x in range(h):
    for y in range(w):
        t1, t2, t3, b1, b2 = 0, 0, 0, 0, 0
        for i in range(5):
            for j in range(5):
                t1 += pad_img[x + i, y + j] * pad_eta[x + i, y + j]
                t2 += pad_img[x + i, y + j]
                t3 += pad_eta[x + i, y + j]
                b1 += pad_eta[x + i, y + j] * pad_eta[x + i, y + j]
                b2 += pad_eta[x + i, y + j]
        t1 /= 25
        t2 /= 25
        t3 /= 25
        b1 /= 25
        b2 /= 25
        if b1 - b2 * b2 == 0:
            weight[x, y] = 1
        else:
            weight[x, y] = (t1 - t2 * t3) / (b1 - b2 * b2)

eta *= weight
plt.imshow(eta, cmap = "gray")
plt.show()

img = np.float64(img) - eta
plt.imshow(img, cmap = "gray")
plt.show()