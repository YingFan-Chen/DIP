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
    tmp = abs(img1 - img2)
    for x in range(h):
        for y in range(w):
            for z in range(c):
                res += tmp[x][y][z] * tmp[x][y][z]
    return res

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

def adjust(img1, img2):
    res = img2
    d = dist(img1, img2)
    for percentage in range(95, 100):
        img_tmp = resize(img2, percentage / 100)
        d_tmp = dist(img1, img_tmp)
        if d > d_tmp:
            res = img_tmp
            d = d_tmp
        # print(d_tmp)
    return res

if __name__ == '__main__':
    input = []
    for i in range(5):
        tmp = imread("./images/" + str(i + 1) + ".jpg")
        input.append(tmp)
    for i in range(5):
        if i == 0:
            save("./images/output" + str(i + 1) + ".jpg", input[i])
        else:
            save("./images/output" + str(i + 1) + ".jpg", adjust(input[0], input[i]))
    