import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./images/keyboard.tif", cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

# (a)
dft_shift = np.fft.fftshift(dft)
dft_magnitude = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
plt.imshow(dft_magnitude, cmap = "gray")
# plt.savefig("test.jpg")
plt.show()

# (b)
