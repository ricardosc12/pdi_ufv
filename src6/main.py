import cv2
from hu import Hu
from svm import Svm
from zernike import Zernike
from utils import preProcessImage, PATH

momento = input("Momento (hu / zernike): ")
if(momento=='hu'):
    print("Executando com momento de Hu \n")
    if PATH == "dataset_gestos":
        hu = Hu(['A','B','C','D','E','F','G','I','L','M','N'])
    else:
        hu = Hu(['acer', 'aerosmith', 'air-jordan', 'airbnb', 'android', 'apple', 'batman', 'bentley', 'boeing', 'boston-celtics', 'chevrolet', 'linux', 'los-angeles-lakers', 'rolling-stones', 'skype'])
    hu.generateHuMoments()
    hu.exportHuMoments()

    svm = Svm("rbf", 100, 1, "dadosHu.csv")
    xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
    yPred = svm.trainModel(xTrain, xTest, yTrain, filename='modelHu_{}.sav'.format(PATH))
    #svm.loadModel('modelHu_{}.sav'.format(PATH))
    svm.getScore(yTest, yPred)

    img = cv2.imread("C1.png", 0)
    imgToPredict = preProcessImage(img)
    svm.predictImage(hu.getHuMoments(imgToPredict))
else:
    zernike = Zernike()
    pca = input("PCA (y / n): ")
    if(pca=="y"):
        print("Executando momento de Zernike com PCA \n")
        zernike.generateZernikeMoments(pca=True)
        zernike.exportZernikeMoments()

        svm = Svm("rbf", 100, 0.1, "dadosZernikePCA.csv")
        xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
        yPred = svm.trainModel(xTrain, xTest, yTrain, filename='modelZernikePCA.sav')
        # svm.loadModel('modelZernikePCA.sav')
    else:
        #TODO: LEMBRAR DE USAR KERNEL 5x5 e NAO INVERTER IMAGEM NA FUNCAO utils.preProcessImage()
        print("Executando momento de Zernike sem PCA \n")
        zernike.generateZernikeMoments(pca=False)
        zernike.exportZernikeMoments()

        svm = Svm("rbf", 100, 50, "dadosZernike.csv")
        xTrain, xTest, yTrain, yTest = svm.splitTrainTest()
        yPred = svm.trainModel(xTrain, xTest, yTrain, filename='modelZernike.sav')
        # svm.loadModel('modelZernike.sav')
    
    svm.getScore(yTest, yPred)

    img = cv2.imread("C1.png", 0)
    imgToPredict = preProcessImage(img)
    svm.predictImage(zernike.getZernikeMoments(imgToPredict))
