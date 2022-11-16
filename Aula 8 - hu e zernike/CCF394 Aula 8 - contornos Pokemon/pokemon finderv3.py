from numpy.linalg import norm
from scipy import misc
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
import numpy as np
import mahotas
import pickle
from sklearn.ensemble import AdaBoostClassifier
import cv2
import numpy as np
import os
import glob
import mahotas
from sklearn.svm import LinearSVC
from sklearn import svm
import argparse
import csv
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from imutils import paths
from matplotlib import pyplot as plt


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def momentosZernike(image):
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("saida",gray)
    #cv2.waitKey(100)

    #aplicando filtro de media 
    filtroMedia = cv2.GaussianBlur(gray,(5,5),0)
 
    ret3,segmentada = cv2.threshold(filtroMedia,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret3,segmentada = cv2.threshold(filtroMedia,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
    #thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,11)
    #thresh = cv2.dilate(thresh, None, iterations=4)
    segmentada = cv2.erode(segmentada, None, iterations=1)
    cv2.imshow("trhesh",segmentada)
    cv2.waitKey(100)

     
    cnts, hierarchy  = cv2.findContours(segmentada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   
    # a função findCountours devolve vários contornos da imagem original
    # existentes devidos a pequenos objetos que ainda aparecem na imagem.
    # este codigo armazena em maior_index o indice do maior contorno obtido
    # e em segunda_index, o segundo maior. O maior contorno tem que ser a mao...
    maior_area = 0
    segunda_area = 0
    maior_index = 0
    segunda_index = 0
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if (area > maior_area):
            if (area > segunda_area):
                segunda_area = maior_area
                maior_area = area
                maior_index = i
        elif (area > segunda_area):
            segunda_area = area
            segunda_index = i
        
    mask = np.zeros(image.shape[:2], dtype="uint8")
    #desenha o maior contorno 
    cv2.drawContours(image, cnts, maior_index, (0, 255, 0), 2)
    cv2.imshow("saida",image)
    cv2.waitKey(100)
    #define o menor retangulo circunscrito no objeto detectado como parametro para recortar o objeto e
    # obter uma imagem destacada do objeto
    (x,y,w,h) = cv2.boundingRect(cnts[maior_index])
    destacada = mask[y:y+h, x:x+w]
    #calcula os monetos e converte para uma array, devolvendo este valor
    caracteristicas = mahotas.features.zernike_moments(segmentada, cv2.minEnclosingCircle(c)[1], degree=8)
    d=np.asarray(caracteristicas)
    return  d



class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius # Size of radius used when computing moments
    
    def describe(self, image): # Returns ZMoments of image
        return mahotas.features.zernike_moments(image, self.radius)
class Searcher:
    # Initializes with an index of sprite Zernike moment features
    # Querying an image's features against it causes it to find the
    # sprite in the database that's nearest
    def __init__(self, index):
        self.index = index
        
    def search(self, queryFeatures):
        results = {}
        for (key, features) in self.index.items():
            distance = norm(queryFeatures - features)
            results[key] = distance
            
        results = sorted([(v, k) for (k, v) in results.items()])
        return results # Nearest sprite is first in list
index = open('pokesprite_index', 'rb')
index = pickle.load(index)
#img = misc.imread('./result_img/ditto.jpg')

img = cv2.imread('pokemon_rb_imgs/ditto.jpg')
cv2.imshow("Pokemon Procurado",img)
cv2.waitKey(0)






queryFeatures=momentosZernike(img)
searcher = Searcher(index) # Initialize db
results = searcher.search(queryFeatures) # Return list of nearest images
print('It\'s a %s!' % results[0][1])

results

