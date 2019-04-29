import cv2 as cv
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv.imread("ce3.jpg", 0)

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

blurred = cv.GaussianBlur(img, (17,17), 0)
g_hpf = img - blurred

cv.imshow("3x3", k3)
cv.imshow("5x5", k5)
cv.imshow("g_hpf", g_hpf)
cv.waitKey()
cv.destroyAllWindows()
