import cv2 as cv
import numpy as np
img=np.zeros((200,200),dtype=np.uint8)
img[50:150, 50:150]=255
ret,thresh=cv.threshold(img,127,255,0)
contours,hierarchy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
color=cv.cvtColor(img,cv.COLOR_GRAY2BGR)
img=cv.drawContours(color,contours,-1,(255,0,0),2)
cv.imshow('cont',color)
cv.waitKey(0)
cv.destroyAllWindows()