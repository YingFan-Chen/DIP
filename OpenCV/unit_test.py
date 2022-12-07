import cv2
import math
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt

def imread(path):
    img = cv2.imread(path)
    img = img[:,:,[2,1,0]]
    return img

def plot(pos, img, title):
    plt.subplot(pos)
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.imshow(img, norm = None)

def show():
    plt.show()

def save(path, img):
    plt.imsave(path, img)

def bilinear(x, y, img):
    l = math.floor(x)
    k = math.floor(y)
    a = x - l
    b = y - k
    res = (1 - a) * (1 - b) * img[l][k] + a * (1 - b) * img[l + 1][k] + (1 - a) * b * img[l][k + 1] + a * b * img[l + 1][k + 1]
    if res > 255:
        res = 255
    if res < 0:
        res = 0
    return round(res)

def resize(img, percentage):
    h, w, c = img.shape
    mid = [h//2, w//2]
    res = np.zeros((h, w, c), dtype="uint8")
    for x in range(h):
        for y in range(w):
            for z in range(c):
                res[x, y, z] = bilinear(percentage * (x - mid[0]) + mid[0], percentage * (y - mid[1]) + mid[1], img[:,:,z])
    return res

if __name__ == '__main__':
    img = imread('./test.jpg')
    res1 = resize(img, 0.95)
    h, w, c = img.shape
    hd, wd = int(h * 0.025), int(w * 0.025)
    tmp = img[hd:h-hd, wd:w-wd]
    res2 = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)
    plot(121, res1, "")
    plot(122, res2, "")
    show()