import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Svm:
    def __init__(self, kernelType, regularization, gamma, data):
        self.df = pd.read_csv(data)
        self.classifier = svm.SVC(kernel=kernelType, C=regularization, gamma=gamma) # rbf, 100, 100
    
    def splitTrainTest(self):
        X = self.df.iloc[:, [i for i in range(len(self.df.columns)-1)]].values
        Y = self.df.iloc[:, len(self.df.columns)-1].values
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, random_state=0)
        return xTrain, xTest, yTrain, yTest
    
    def trainModel(self, xTrain, xTest, yTrain):
        print("Treinando o modelo...")
        yPred = self.classifier.fit(xTrain, yTrain).predict(xTest)
        return yPred
    
    def getScore(self, yTest, yPred):
        tempYTest = np.array(yTest)
        tempYPred = np.array(yPred)
        np.set_printoptions(precision=2)

        precision = 100 * accuracy_score(tempYTest, tempYPred)
        print(f'Precisao do modelo: {precision:.3f} %')
        self.getConfusionMatrix(tempYTest, tempYPred)
    
    def getConfusionMatrix(self, yTest, yPred):
        # TODO
        pass
    
    def predictImage(self, data):
        print("Predizendo imagem...")
        output = self.classifier.predict([data])
        self.translateOutput(output)
    
    def translateOutput(self, output):
        if output == 0:
            print("Esta imagem e uma letra A!")
        elif output == 1:
            print("Esta imagem e uma letra B!")
        elif output == 2:
            print("Esta imagem e uma letra C!")
        elif output == 3:
            print("Esta imagem e uma letra D!")
        elif output == 4:
            print("Esta imagem e uma letra E!")
        elif output == 5:
            print("Esta imagem e uma letra F!")
        elif output == 6:
            print("Esta imagem e uma letra G!")
        elif output == 7:
            print("Esta imagem e uma letra I!")
        elif output == 8:
            print("Esta imagem e uma letra L!")
        elif output == 9:
            print("Esta imagem e uma letra M!")
        elif output == 10:
            print("Esta imagem e uma letra N!")
