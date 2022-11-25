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

# gamma : (0 ~ 1) -> light
# gamma : (1 ~ inf) -> dark
def gammafunc(img, gamma = 1, channel = [0,1,2], A = 1):
    res = np.copy(img)
    res = np.float64(res)
    for ch in channel:
        res[:, :, ch] = (A * (res[:, :, ch] / 255) ** gamma) * 255
    res = np.uint8(res)
    return res

def sigmoidfun(img, channel = [0,1,2]):
    res = np.copy(img)
    res = np.float64(res)
    for ch in channel:
        res[:, :, ch] = (res[:, :, ch] * 10 / 255) - 5
        res[:, :, ch] = 255 / (1 + math.e ** (- res[:, :, ch]))
    res = np.uint8(res)
    return res

if __name__ == '__main__':
    a = [1,2,3]
    for x in a:
        x = 5
    print(a)