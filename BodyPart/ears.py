import cv2 as cv
import numpy as np

left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')

cap = cv.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)
    right_ear = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)

    for (x, y, w, h) in left_ear:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (x, y, w, h) in right_ear:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow('Ear Detector', frame)
    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()