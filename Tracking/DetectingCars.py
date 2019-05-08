import cv2 as cv
import numpy as np
from os.path import join

datapath = "../Tracking/CarData/TrainImages/"

def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i + 1)

pos, neg = "pos-", "neg-"

detect = cv.SIFT_create()
extract = cv.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
matcher = cv.FlannBasedMatcher(flann_params, {})


bow_kmeans_trainer = cv.BOWKMeansTrainer(40)
extract_bow = cv.BOWImgDescriptorExtractor(extract, matcher)




def extract_sift(fn):
    im = cv.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


def bow_features(fn):
    im = cv.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))


traindata, trainlabels = [], []
for i in range(20):
    traindata.extend(bow_features(path(pos, i)));
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg, i)));
    trainlabels.append(-1)

svm = cv.ml.SVM_create()
svm.train(np.array(traindata), cv.ml.ROW_SAMPLE, np.array(trainlabels))




def predict(fn):
    f = bow_features(fn);
    p = svm.predict(f)
    print (fn, "\t", p[1][0][0])
    return p

car, notcar = "../Tracking/car.jpg", "../Tracking/not_car.jpg"
car_img = cv.imread(car)
notcar_img = cv.imread(notcar)

car_predict = predict(car)
not_car_predict = predict(notcar)

font = cv.FONT_HERSHEY_SIMPLEX

if (car_predict[1][0][0] == 1.0):
    cv.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv.LINE_AA)

if (not_car_predict[1][0][0] == -1.0):
    cv.putText(notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv.LINE_AA)

cv.imshow('BOW + SVM Success', car_img)
cv.imshow('BOW + SVM Failure', notcar_img)
cv.waitKey(0)
cv.destroyAllWindows()
