import numpy as np

# img1, img2 -> 2D-array with value 0 - 255
def Add(img1, img2):
    h = np.size(img1, 0)
    w = np.size(img1, 1)
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            if img1[i][j] + img2[i][j] > 255:
                res[i][j] = 255
            else:
                res[i][j] = img1[i][j] + img2[i][j]

# img1, img2 -> 2D-array with value 0 - 255
# img1 - img2
def Subtract(img1, img2):
    h = np.size(img1, 0)
    w = np.size(img1, 1)
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            if img1[i][j] - img2[i][j] < 0:
                res[i][j] = 0
            else:
                res[i][j] = img1[i][j] - img2[i][j]

# img -> 2D-array with value 0 - 255, bit -> 0 - 7
def BitPlane(img, bit):
    h = np.size(img, 0)
    w = np.size(img, 1)
    res = np.zeros((h, w), dtype = "uint8")
    mask = 1 << bit
    for i in range(h):
        for j in range(w):
            res[i][j] = img[i][j] & mask
    return res

# img -> 2D-array with value 0 - 255, gamma -> float point
# Assume c = 1
def GammaTransformation(img, gamma):
    h = np.size(img, 0)
    w = np.size(img, 1)
    res = np.zeros((h, w), dtype = "uint8")
    for i in range(h):
        for j in range(w):
            res[i][j] = round(((img[i][j] / 255) ** gamma) * 255)
    return res

# img -> 2D-array with value 0 - 255, pz -> 1D-array with float point and pz.size() == 255
# pz is the probability we want
def HistogramMatch(img, pz):
    h = np.size(img, 0)
    w = np.size(img, 1)
    count = np.zeros(256)
    for i in range(h):
        for j in range(w):
            count[img[i][j]] += 1
    
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
            res[i][j] = match[img[i][j]]
    return res
    