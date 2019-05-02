import cv2

img = cv2.imread('ce3.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(float(8000))
kp, des = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)
cv2.imshow('SURF features', img)
cv2.waitKey()