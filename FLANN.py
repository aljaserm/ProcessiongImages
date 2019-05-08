import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

queryImage = cv.imread('input2.png',0)
trainingImage = cv.imread('input1.png',0)

# create SIFT and detect/compute
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(queryImage,None)
kp2, des2 = sift.detectAndCompute(trainingImage,None)

# FLANN matcher parameters
# FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = 0, trees = 5)
searchParams = dict(checks=20)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(indexParams,searchParams)

matches = flann.knnMatch(des1,des2,k=2)

# prepare an empty mask to draw good matches
matchesMask = [[0,0] for i in range(len(matches))]

# David G. Lowe's ratio test, populate the mask
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

drawParams = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

resultImage = cv.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams)

plt.imshow(resultImage,),plt.show()