import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret,frame = cap.read()
    frame = cv.resize(frame, None, fx=scaling_factor,fy=scaling_factor, interpolation=cv.INTER_AREA)

    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)


    for (x,y,w,h) in face_rects:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv.imshow("Face Detection", frame)

    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()
cv.destroyAllWindows()
