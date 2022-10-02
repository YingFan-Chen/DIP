from cv2 import resize
import lib
import cv2
import math
import numpy as np

def Padding(img, p = 0):
    h, w = img.shape
    res = np.zeros((h + 2 * p, w + 2 * p), dtype = "uint8")
    res[p : h + p, p : w + p] = img
    
    res[p : h + p, 0 : p] = 0
    res[p : h + p, w + p : w + 2 * p] = 0
    res[0 : p, p : w + p] = 0
    res[h + p : h + 2 * p, p : w + p] = 0

    res[0 : p, 0 : p] = 0
    res[0 : p, h + p : h + 2 * p] = 0
    res[h + p : h + 2 * p, 0 : p] = 0
    res[h + p : h + 2 * p, w + p : w + 2 * p] = 0
    return res

def Resize(img, h, w):
    origin_h, origin_w = img.shape
    h = round(h)
    w = round(w)
    res = np.zeros((h, w), dtype = "uint8")
    h_rate = origin_h / h
    w_rate = origin_w / w
    for i in range(h):
        for j in range(w):
            pad = Padding(img, 2)
            res[i, j] = Bicubic(i * h_rate + 2, j * w_rate + 2, pad)
    return res

def Bicubic(x, y, img, a = -0.5):
    def W(x_, a_):
        x_ = abs(x_)
        if x_ <= 1:
            return (a_ + 2) * (x_ ** 3) - (a_ + 3) * (x_ ** 2) + 1
        elif x < 2:
            return a_ * (x_ ** 3) - 5 * a_ * (x_ ** 2) + 8 * a_ * x_ - 4 * a_
        else:
            return 0
    l = math.floor(x)
    k = math.floor(y)
    res = 0
    for i in range(l - 1, l + 3, 1):
        for j in range(k - 1, k + 3, 1):        
            res += img[i, j] * W(x - i, a) * W(y - j, a)
    if res > 255:
        res = 255
    if res < 0:
        res = 0
    return round(res)

img = cv2.imread("T.png", cv2.IMREAD_GRAYSCALE)
img = Resize(img, round(0.77 * 256), round(0.77 * 256))
cv2.imshow("img", img)
cv2.waitKey(0)
