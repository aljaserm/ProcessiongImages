import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('input2.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('input1.png',cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
# matches = bf.knnMatch(des1,des2, k=2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1,kp1,img2,kp2, matches[:40],img2,flags=2)

plt.imshow(img3),plt.show()
