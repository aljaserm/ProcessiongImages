import cv2 as cv

gray_image = cv.imread('input.jpg', 0)

fast = cv.FastFeatureDetector_create()

brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(gray_image, None)

keypoints, descriptors = brief.compute(gray_image, keypoints)
gray_keypoints = cv.drawKeypoints(image=gray_image,outImage=gray_image, keypoints=keypoints,color=(0,255,0))

cv.imshow('BRIEF keypoints', gray_keypoints)
cv.waitKey()