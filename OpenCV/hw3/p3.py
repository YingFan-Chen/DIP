import cv2
import math
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt

def imread(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def plot(pos, img, title):
    plt.subplot(pos)
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.imshow(img, cmap = 'gray')

def show():
    plt.show()

def save(path, img):
    plt.imsave(path, img)

def padding(val, pad, img):
    h, w = img.shape
    padding_img = np.arange((h + 2*pad, w + 2*pad), val)
    padding_img[pad:pad+h, pad:pad+w] = img
    return padding_img

if __name__ == '__main__':
    img = imread('./images/bricks.tif')
    plot(121, img, 'Origin')
    padding_img = padding(0, 1, img)
    kernel = np.array([
        [255, 255, 255],
        [255. 255, 255],
        [255, 255, 255]
    ])