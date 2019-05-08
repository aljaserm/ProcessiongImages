import cv2 as cv
import numpy as np

nose_cascade = cv.CascadeClassifier('haarcascade_mcs_nose.xml')

cap = cv.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in nose_rects:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        break

    cv.imshow('Nose Detector', frame)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()