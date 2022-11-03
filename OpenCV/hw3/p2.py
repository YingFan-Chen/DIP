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

def regionAVG(img, x_range, y_range):
    res = np.zeros(3, dtype='float64')
    for ch in range(3):
        res[ch] = np.sum(img[x_range[0]:x_range[1],y_range[0]:y_range[1],ch])
    divide = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    res = res / divide
    res = np.uint8(res)
    return res

def colorSegment(img, color, radius = 10):
    h, w, c = img.shape
    res = np.zeros((h, w, c), dtype='uint8')
    for x in range(h):
        for y in range(w):
            if math.dist(img[x , y, :], color) <= radius:
                res[x, y, :] = [255, 255, 255]
    return res

if __name__ == '__main__':
    img = imread('./images/20-2851.tif')
    plot(121, img, 'Origin')
    x_range, y_range = [730, 830], [423, 523]
    color = regionAVG(img, x_range, y_range)
    segmentation = colorSegment(img, color, 35)
    plot(122, segmentation, 'Segmentation')
    show()
    save('./images/20-2851_segmentation.jpg', segmentation)