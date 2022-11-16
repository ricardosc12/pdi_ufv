import cv2
from Hu import Hu
from Svm import Svm
from utils import preProcessImage

hu = Hu()
hu.generateHuMoments()
hu.exportHuMoments()

svm = Svm("rbf", 100, 100, "dadosHu.csv")
xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
yPred = svm.trainModel(xTrain, xTest, yTrain)
svm.getScore(yTest, yPred)

img = cv2.imread("G.png", 0)
imgToPredict = preProcessImage(img)
svm.predictImage(hu.getHuMoments(imgToPredict))
