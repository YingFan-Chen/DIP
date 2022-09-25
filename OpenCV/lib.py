import numpy as np
import math

Nearest = 0
Linear = 1
Cubic = 2

# p.range = 0 ~ min(h - 1, w - 1)
# This is a mirror padding.
def Padding(img, p = 0):
    h, w = img.shape
    res = np.zeros((h + 2 * p, w + 2 * p), dtype = "uint8")
    res[p : h + p, p : w + p] = img
    
    res[p : h + p, 0 : p] = img[0 : h, p - 1 : : -1]
    res[p : h + p, w + p : w + 2 * p] = img[0 : h, w - 1 : w - 1 - p : -1]
    res[0 : p, p : w + p] = img[p - 1 : : -1, 0 : w]
    res[h + p : h + 2 * p, p : w + p] = img[h - 1 : h - 1 - p : -1, 0 : w]

    res[0 : p, 0 : p] = img[p - 1: : -1, p - 1 : : -1]
    res[0 : p, h + p : h + 2 * p] = img[p - 1 : : -1, w - 1 : w - p - 1 : -1]
    res[h + p : h + 2 * p, 0 : p] = img[h - 1: h - 1 - p : -1, p - 1 : : -1]
    res[h + p : h + 2 * p, w + p : w + 2 * p] = img[h - 1 : h - 1 - p : -1, w - 1 : w - 1 - p : -1]
    return res

def Nearest_Neighbor(x, y, img):
    x = round(x)
    y = round(y)
    return img[x, y]

def Bilinear(x, y, img):
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

def Correlation(img, kernal, x, y):
    kernal_h, kernal_w = kernal.shape
    res = 0
    for i in range(kernal_h):
        for j in range(kernal_w):
            res += img[x + i, y + j] * kernal[i, j]
    if res > 255:
        res = 255
    if res < 0:
        res = 0
    return round(res)

# img1, img2 -> 2D-array with value 0 - 255
def Add(img1, img2):
    h, w = img1.shape
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            if img1[i, j] + img2[i, j] > 255:
                res[i, j] = 255
            else:
                res[i, j] = img1[i, j] + img2[i, j]
    return res

# img1, img2 -> 2D-array with value 0 - 255
# img1 - img2
def Subtract(img1, img2):
    h, w = img1.shape
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            if img1[i, j] - img2[i, j] < 0:
                res[i, j] = 0
            else:
                res[i, j] = img1[i, j] - img2[i, j]
    return res

def Multiple(img1, img2):
    h, w = img1.shape
    res = np.zeros((h, w), dtype="uint8")
    for i in range(h):
        for j in range(w):
            if img1[i, j] * img2[i, j] > 255:
                res[i, j] = 255
            else:
                res[i, j] = round(img1[i, j] * img2[i, j])
    return res

def Divide(img1, img2):
    h, w = img1.shape
    res = np.zeros((h,w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            if img2[i, j] != 0:
                res[i, j] = round(img1[i, j] / img2[i, j])
            else:
                res[i, j] = round(img1[i, j])
    return res

# img -> 2D-array with value 0 - 255, bit -> 0 - 7
def BitPlane(img, bit):
    h, w = img.shape
    res = np.zeros((h, w), dtype = "uint8")
    mask = 1 << bit
    for i in range(h):
        for j in range(w):
            res[i, j] = img[i, j] & mask
    return res

# img -> 2D-array with value 0 - 255, gamma -> float point
def GammaTransformation(img, gamma, c = 1):
    h, w = img.shape
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            res[i, j] = round(((img[i, j] / 255) ** gamma) * 255 * c)
    return res

# img -> 2D-array with value 0 - 255, pz -> 1D-array with float point and pz.size() == 255
# pz is the probability we want.
def HistogramMatch(img, pz):
    h, w = img.shape
    count = np.zeros(256)
    for i in range(h):
        for j in range(w):
            count[img[i, j]] += 1
    
    s = np.zeros(256)
    cul = 0
    total = h * w
    for i in range(256):
        cul += count[i]
        s[i] = round(255 * cul / total)
    
    g = np.zeros(256)
    cul = 0
    for i in range(256):
        cul += pz[i]
        g[i] = round(255 * cul)

    match = np.zeros(256)
    r = 255
    z = 255
    while r >= 0 and z >= 0:
        if s[r] < g[z]:
            z -= 1
        elif s[r] == g[z]:
            if g[z] == g[z - 1]:
                z -= 1
            else:
                match[r] = z
                r -= 1
        else:
            match[r] = z
            r -= 1
    
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            res[i, j] = match[img[i, j]]
    return res

# h, w -> new size
def Resize(img, h, w, interpolation = Nearest):
    origin_h, origin_w = img.shape
    h = round(h)
    w = round(w)
    res = np.zeros((h, w), dtype = "uint8")
    h_rate = origin_h / h
    w_rate = origin_w / w
    for i in range(h):
        for j in range(w):
            if interpolation == 0:
                pad = Padding(img, 1)
                res[i, j] = Nearest_Neighbor(i * h_rate + 1, j * w_rate + 1, pad)
            elif interpolation == 1:
                pad = Padding(img, 1)
                res[i, j] = Bilinear(i * h_rate + 1, j * w_rate + 1, pad)
            elif interpolation == 2:
                pad = Padding(img, 2)
                res[i, j] = Bicubic(i * h_rate + 2, j * w_rate + 2, pad)
    return res

# Theta is angle(360), h, w -> new size
# Clockwise Rotate
def Rotate(img, h, w, theta, interpolation = Nearest):
    theta %= 360
    theta = (theta + 360) % 360
    phi = theta * math.pi / 180
    img_h, img_w = img.shape

    res = np.zeros((h, w), dtype = "uint8")
    cos = math.cos(phi)
    sin = math.sin(phi)
    origin_x = math.floor(h / 2)
    origin_y = math.floor(w / 2)
    img_origin_x = math.floor(img_h / 2)
    img_origin_y = math.floor(img_w / 2)
    for i in range(h):
        for j in range(w):
            x = cos * (i - origin_x) - sin * (j - origin_y) + img_origin_x
            y = sin * (i - origin_x) + cos * (j - origin_y) + img_origin_y
            if x >= 0 and x < img_h and y >= 0 and y < img_w:
                if interpolation == 0:
                    pad = Padding(img, 1)
                    res[i, j] = Nearest_Neighbor(x + 1, y + 1, pad)
                elif interpolation == 1:
                    pad = Padding(img, 1)
                    res[i, j] = Bilinear(x + 1, y + 1, pad)
                elif interpolation == 2:
                    pad = Padding(img, 2)
                    res[i, j] = Bicubic(x + 2, y + 2, pad)
            else:
                res[i, j] = 255
    return res
    
def GaussainFiliter(img, size, sigma = 1, K = 1):
    h, w = img.shape
    if size % 2 == 0:
        size += 1
    kernal = np.zeros((size, size))
    mid = math.floor(size / 2)
    sum = 0
    for i in range(size):
        for j in range(size):
            kernal[i, j] = K * math.e ** (((i - mid) ** 2 + (j - mid) ** 2) / (2 * (sigma ** 2)))
            sum += kernal[i, j]
    for i in range(size):
        for j in range(size):
            kernal[i, j] /= sum
    pad = Padding(img, size - 1)
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            res[i, j] = Correlation(pad, kernal, i, j)
        print("(", i, "/", h, ")")
    return res  