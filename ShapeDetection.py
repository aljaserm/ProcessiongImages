import cv2 as cv
import numpy as np

planets = cv.imread('ce3.jpg')
gray_img = cv.cvtColor(planets, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(gray_img, 5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,120,
                            param1=100,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv.circle(planets,(i[0],i[1]),i[2],(0,255,0),2)

    cv.circle(planets,(i[0],i[1]),2,(0,0,255),3)

cv.imwrite("planets_circles.jpg", planets)
cv.imshow("HoughCirlces", planets)
cv.waitKey()
cv.destroyAllWindows()

# cv.findContours
# cv.approxPolyDP