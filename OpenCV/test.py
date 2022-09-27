import lib
import cv2
import numpy as np

img = cv2.imread("tmp.jpg")
img = cv2.resize(img, (512, 512))
res = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
res[:,:,2] = cv2.equalizeHist(res[:,:,2])
res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
cv2.imshow("res", res)
cv2.imshow("img", img)
cv2.waitKey(0)