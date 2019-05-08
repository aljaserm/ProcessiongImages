import cv2 as cv

img = cv.imread('ce3.jpg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

surf = cv.xfeatures2d.SURF_create(float(8000))
kp, des = surf.detectAndCompute(gray, None)

img = cv.drawKeypoints(img, kp, None, (0,255,0), 4)
cv.imshow('SURF features', img)
cv.waitKey()