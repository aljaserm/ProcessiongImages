import cv2
import numpy as np

img = cv2.imread('ce3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input', img)

corner = cv2.goodFeaturesToTrack(gray, maxCorners=7, qualityLevel=0.05, minDistance=25)
corner = np.float32(corner)

for item in corner:
    x,y=item[0]
    cv2.circle(img,(x,y),5,255,-1)
cv2.imshow('Top k feature', img)
cv2.waitKey()
