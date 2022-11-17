import cv2 # Biblioteca OpenCV
import pandas as pd # Biblioteca pandas para gerar o csv
from sklearn.preprocessing import MinMaxScaler # Biblioteca para transformar as features em um range
from utils import * # Modulo com algumas funcoes auxiliares
from math import copysign, log10
import mahotas
from sklearn.decomposition import PCA
import numpy as np

class Zernike:
    def __init__(self):
        self.xMoments = [] # Armazena os momentos de hu
        self.yMoments = [] # Armazena de qual letra cada momento de hu e
        self.letters = ['A','B','C','D','E','F','G','I','L','M','N'] # Letras do dataset
        self.pca = False
    
    def getZernikeMoments(self,image):
        momentos = mahotas.features.zernike_moments(image, 8)
        if(self.pca):
            pca = PCA(n_components=0.9, whiten=True)
            momentos = np.array(momentos)
            momentos = pca.fit_transform(momentos.reshape(-1,1)).flatten()
        return momentos

    def generateZernikeMoments(self,pca=False):
        print("Gerando zernike momentos..." if not pca else "Gerando zernike moments com pca...")
        count = 0
        self.pca = pca
        for letter in self.letters:
            files = getImagesByLetter(letter)
            for file in files:
                image = getImageFromDataset(letter, file)
                self.xMoments.append(self.getZernikeMoments(image))
                self.yMoments.append(count)
                # cv2.imshow("asd", image)
                # cv2.waitKey(0)
                # break
            count += 1
        # exit(1)
        

    def exportZernikeMoments(self):
        print("Exportando dados...")
        columns = []
        for i in range(len(self.xMoments[0])):
            columns.append('M{}'.format(i))
        df = pd.DataFrame(self.xMoments, columns = columns)
        dfFinal = df
        dfFinal['Y'] = self.yMoments
        dfFinal.to_csv("dadosZernike.csv", header=False, index=False)
