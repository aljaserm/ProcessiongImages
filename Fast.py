import cv2
import numpy as np

input_image = cv2.imread('ce3.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()

keypoints = fast.detect(gray_image, None)
print ("Number of keypoints with non max suppression:", len(keypoints))

img_keypoints_with_nonmax = input_image.copy()
result=cv2.drawKeypoints(image=gray_image, outImage=gray_image,keypoints=keypoints, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST keypoints - with non max suppression', result)

keypoints = fast.detect(gray_image, None)

print ("Total Keypoints without nonmaxSuppression:", len(keypoints))

img_keypoints_without_nonmax = input_image.copy()
result1=cv2.drawKeypoints(image=gray_image, outImage=gray_image, keypoints=keypoints, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST keypoints - without non max suppression', result1)
cv2.waitKey()