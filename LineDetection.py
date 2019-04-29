import cv2 as cv
import numpy as np
img=cv.imread('ce3.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges=cv.Canny(gray,50,120)
minLine=20
maxLine=5
line=cv.HoughLinesP(edges,1,np.pi/180,20,minLine,maxLine)
for x,y,x1,y1 in line[0]:
    cv.line(img,(x,y),(x1,y1),(255,0,0),2)

cv.imshow('edge',edges)
cv.imshow('line',img)
cv.waitKey(0)
cv.destroyAllWindows()