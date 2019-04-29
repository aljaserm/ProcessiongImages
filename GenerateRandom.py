import cv2
import numpy
import os

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

#convert to 400*300
grayImage = flatNumpyArray.reshape(300,400)
cv2.imwrite('randomgray.png',grayImage)

#convert to 400*100
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)

numpy.random.randint(0,256,120000).reshape(300,400)