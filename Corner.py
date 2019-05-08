import cv2 as cv
import numpy as np

img = cv.imread('ce3.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Input', img)

corner = cv.goodFeaturesToTrack(gray, maxCorners=7, qualityLevel=0.05, minDistance=25)
corner = np.float32(corner)

for item in corner:
    x,y=item[0]
    cv.circle(img,(x,y),5,255,-1)
cv.imshow('Top k feature', img)
cv.waitKey()
