import cv2
import math
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
    plt.imshow(img, vmin = 0, vmax=255)

def show():
    plt.show()

def save(path, img):
    plt.imsave(path, img)

def dist(img1, img2):
    assert(img1.shape == img2.shape)
    h, w, c = img1.shape
    res = 0
    tmp = img1 - img2
    tmp = np.multiply(tmp, tmp)
    res = np.sum(tmp)
    return res

def resize(img, percentage):
    h, w, c = img.shape
    hd, wd = int(h * (1 - percentage) / 2), int(w * (1 - percentage) / 2)
    tmp = img[hd:h-hd, wd:w-wd]
    res = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_CUBIC)
    return res

def adjust(img1, img2):
    res = img2
    d = dist(img1, img2)
    for percentage in range(900, 1000):
        img_tmp = resize(img2, percentage / 1000)
        d_tmp = dist(img1, img_tmp)
        if d > d_tmp:
            res = img_tmp
            d = d_tmp
        # print(d_tmp)
    return res

def f(img1, img2, t):
    print(img1.shape)
    h, w, c = img1.shape
    res = np.zeros((h, w, c), dtype='uint8')
    for x in range(h):
        for y in range(w):
            for z in range(c):
                res[x, y, z] = round(t * img1[x, y, z] + (1 - t) * img2[x, y, z])
    return res

if __name__ == '__main__':
    img1 = imread('./images/output1.jpg')
    img2 = imread('./images/output2.jpg')
    img3 = imread('./images/output3.jpg')
    print(img1.shape)
    for i in range(10):
        t = i / 10
        res = img1 * t + img3 * (1 - t)
        res = np.uint8(res)
        plot(121, res, 'mid')
        plot(122, img2, '2')
        show()