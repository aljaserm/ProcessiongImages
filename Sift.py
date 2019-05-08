import cv2 as cv

input_image = cv.imread('ce3.jpg')
gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray_image, None)

input_image = cv.drawKeypoints(image=input_image, outImage=input_image, keypoints=keypoints,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color = (51, 163, 236))

cv.imshow('SIFT features', input_image)
cv.waitKey()