import cv2 as cv
import numpy as np

mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')

cap = cv.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=11)
    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        break

    cv.imshow('Mouth Detector', frame)

    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()