import cv2
from hu import Hu
from svm import Svm
from zernike import Zernike
from utils import preProcessImage
import mahotas
from os import listdir
from os.path import isfile, join

# HU: 100 1
# Zernike PCA 100 0.1
# Zernike 100 50            img sem inverter e blur 5,5

# zernike = Zernike()
# zernike.generateZernikeMoments(pca=False)
# zernike.exportZernikeMoments()

# svm = Svm("rbf", 100, 50, "dadosZernike.csv")
# xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
# yPred = svm.trainModel(xTrain, xTest, yTrain,filename='modelZernike.sav')
# # svm.loadModel('modelZernike.sav')
# svm.getScore(yTest, yPred)

# img = cv2.imread("C1.png", 0)
# imgToPredict = preProcessImage(img)

# svm.predictImage(zernike.getZernikeMoments(imgToPredict))



# HU 
hu = Hu(['A','B','C','D','E','F','G','I','L','M','N'])
# hu = Hu(['acer', 'aerosmith', 'air-jordan', 'airbnb', 'android', 'apple', 'batman', 'bentley', 'boeing', 'boston-celtics', 'chevrolet', 'linux', 'los-angeles-lakers', 'rolling-stones', 'skype'])
# hu.generateHuMoments()
# hu.exportHuMoments()

svm = Svm("rbf", 100, 1, "dadosHu.csv")
# xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
# yPred = svm.trainModel(xTrain, xTest, yTrain,filename='modelHu.sav')
svm.loadModel('perfeitinha2.sav')
# svm.getScore(yTest, yPred)

img = cv2.imread("C1.png", 0)
imgToPredict = preProcessImage(img)
# cv2.imshow("asd",imgToPredict)
# cv2.waitKey(0)
svm.predictImage(hu.getHuMoments(imgToPredict))
