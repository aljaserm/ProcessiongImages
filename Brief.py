import cv2

gray_image = cv2.imread('input.jpg', 0)

fast = cv2.FastFeatureDetector_create()

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(gray_image, None)

keypoints, descriptors = brief.compute(gray_image, keypoints)
gray_keypoints = cv2.drawKeypoints(image=gray_image,outImage=gray_image, keypoints=keypoints,color=(0,255,0))

cv2.imshow('BRIEF keypoints', gray_keypoints)
cv2.waitKey()