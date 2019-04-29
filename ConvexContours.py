import cv2 as cv
import numpy as np

img = cv.pyrDown(cv.imread("ce3.jpg", cv.IMREAD_UNCHANGED))

ret, thresh = cv.threshold(cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY) , 127, 255, cv.THRESH_BINARY)
black = cv.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv.COLOR_GRAY2BGR)

contours, hier = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
  epsilon = 0.01 * cv.arcLength(cnt,True)
  approx = cv.approxPolyDP(cnt,epsilon,True)
  hull = cv.convexHull(cnt)
  cv.drawContours(black, [cnt], -1, (0, 255, 0), 2)
  cv.drawContours(black, [approx], -1, (255, 255, 0), 2)
  cv.drawContours(black, [hull], -1, (0, 0, 255), 2)

cv.imshow("hull", black)
cv.waitKey()
cv.destroyAllWindows()
