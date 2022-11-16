import cv2 # Biblioteca OpenCV
import pandas as pd # Biblioteca pandas para gerar o csv
from sklearn.preprocessing import MinMaxScaler # Biblioteca para transformar as features em um range
from utils import * # Modulo com algumas funcoes auxiliares

class Hu:
    def __init__(self):
        self.xMoments = [] # Armazena os momentos de hu
        self.yMoments = [] # Armazena de qual letra cada momento de hu e
        self.letters = ['A','B','C','D','E','F','G','I','L','M','N'] # Letras do dataset
        self.scaler = MinMaxScaler() # MinMax para transformar os momentos de Hu
    
    def getHuMoments(self, image):
        huMoments = cv2.HuMoments(cv2.moments(image)).flatten()
        return huMoments

    def generateHuMoments(self):
        print("Gerando momentos...")
        count = 0
        for letter in self.letters:
            files = getImagesByLetter(letter)
            for file in files:
                image = getImageFromDataset(letter, file)
                self.xMoments.append(self.getHuMoments(image))
                self.yMoments.append(count)
            count += 1
    
    def transformHuData(self, df):
        for col in df.columns:
            df[col] = minMaxScaling(df[col])

        self.scaler.fit(df)
        scaled = self.scaler.fit_transform(df)
        scaledMoments = pd.DataFrame(scaled, columns=df.columns)
        dfFinal = scaledMoments.round(decimals=2)
        return dfFinal

    def exportHuMoments(self):
        print("Exportando dados...")
        df = pd.DataFrame(self.xMoments, columns = ['M1','M2','M3','M4','M5','M6','M7'])
        dfFinal = self.transformHuData(df)
        dfFinal['Y'] = self.yMoments
        dfFinal.to_csv("dadosHu.csv", header=False, index=False)
