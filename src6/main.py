import cv2
from hu import Hu
from svm import Svm
from utils import preProcessImage
import mahotas

hu = Hu()
# hu.generateHuMoments()
# hu.exportHuMoments()

svm = Svm("rbf", 100, 1, "dadosHu.csv")
# xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
# yPred = svm.trainModel(xTrain, xTest, yTrain)
svm.loadModel()
# svm.getScore(yTest, yPred)

img = cv2.imread("C1.png", 0)
imgToPredict = preProcessImage(img)
# cv2.imshow("asd", imgToPredict)
# cv2.waitKey(0)
svm.predictImage(hu.getHuMoments(imgToPredict))
