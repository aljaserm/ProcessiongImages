import cv2 as cv
import numpy as np
from Tracking import DetectingCars
from Tracking import pyramid
from Tracking import nonmaximum as nms
from Tracking import slidingwindow
import urllib.request


def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh


# test_image = "../cars.jpg"
# img_path = "../test.jpg"
#
# urllib.request.urlretrieve(test_image, img_path)

# test_image = cv.imread("../car.jpg")
# cv.imshow("test",test_image)
svm, extractor = DetectingCars()
detect = cv.SIFT_create()

w, h = 100, 40
img = cv.imread("../car7.jpg")

rectangles = []
counter = 1
scaleFactor = 1.25
scale = 1
font = cv.FONT_HERSHEY_PLAIN

for resized in pyramid(img, scaleFactor):
    scale = float(img.shape[1]) / float(resized.shape[1])
    for (x, y, roi) in slidingwindow(resized, 20, (100, 40)):

        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            bf = DetectingCars.bowFeatures(roi, extractor, detect)
            _, result = svm.predict(bf)
            a, res = svm.predict(bf, flags=cv.ml.STAT_MODEL_RAW_OUTPUT | cv.ml.STAT_MODEL_UPDATE_MODEL)
            print ("Class: %d, Score: %f, a: %s" % (result[0][0], res[0][0], res))
            score = res[0][0]
            if result[0][0] == 1:
                if score < -1.0:
                    rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                    rectangles.append([rx, ry, rx2, ry2, abs(score)])
        except:
            pass

        counter += 1

windows = np.array(rectangles)
boxes = nms(windows, 0.25)

for (x, y, x2, y2, score) in boxes:
    print (x, y, x2, y2, score)
    cv.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv.putText(img, "%f" % score, (int(x), int(y)), font, 1, (0, 255, 0))

cv.imshow("img", img)
cv.waitKey(0)
