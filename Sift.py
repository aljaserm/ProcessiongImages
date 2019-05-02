import cv2

input_image = cv2.imread('ce3.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray_image, None)

input_image = cv2.drawKeypoints(image=input_image, outImage=input_image, keypoints=keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color = (51, 163, 236))

cv2.imshow('SIFT features', input_image)
cv2.waitKey()