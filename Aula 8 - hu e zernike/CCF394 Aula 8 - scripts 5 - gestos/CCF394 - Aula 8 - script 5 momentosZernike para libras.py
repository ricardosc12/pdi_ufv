## https://gogul09.github.io/software/texture-recognition

#https://www.learnopencv.com/opencv-threshold-python-cpp/

from sklearn.ensemble import AdaBoostClassifier
import cv2
import numpy as np
import os
import glob
import mahotas
import mahotas.demos
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


def chamaPredicao(imagem):
    if imagem.sum()>300:
        momentos = momentosZernike(imagem)
        caracteristicas=np.asarray(momentos)
        caracteristicas=(momentos)
        predicao = clf_svm.predict(np.asarray(caracteristicas).reshape(1,-1))[0]
        print("Letra ->  ",predicao[0])
        cv2.putText(imagem, predicao[0], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        cv2.namedWindow('Imagem Teste',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',  round(imagem.shape[1]/2), round(imagem.shape[2]/2))
        cv2.imshow("Imagem Teste", imagem)
        cv2.waitKey(100)

# funcao que extrai os momentos de zernike de contornos de imagens

def momentosZernike(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #aplicando filtro de media 
    filtroMedia = cv2.GaussianBlur(gray,(5,5),0)
    ret3,segmentada = cv2.threshold(filtroMedia,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    segmentada = cv2.erode(segmentada, None, iterations=1)
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
    #cv2.drawContours(image, cnts, maior_index, (0, 255, 0), 2)
    #cv2.imshow("saida",image)
    #cv2.waitKey(100)
    #define o menor retangulo circunscrito no objeto detectado como parametro para recortar o objeto e
    # obter uma imagem destacada do objeto
    (x,y,w,h) = cv2.boundingRect(cnts[maior_index])
    destacada = mask[y:y+h, x:x+w]
    #calcula os monetos e converte para uma array, devolvendo este valor
    caracteristicas =mahotas.features.zernike_moments(segmentada, cv2.minEnclosingCircle(c)[1], degree=8)
    d=np.asarray(caracteristicas)
    return  d

# codigo que le imagens do diretorio de treino e monta um classificador...
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
print( "[STATUS]Extraindo caracteristicas ... ")
dados = []
labels = []
saidaX=[]
saidaLabel=[]
# define diretorio para imagens de treino e de teste
args["treinando"]="gestos\\treino"
args["testando"]="gestos\\teste"
print('teste')
print(args["testando"])
print(paths.list_images(args["treinando"]))
# loop lendo as imagens de treinamento
for imagePath in paths.list_images(args["treinando"]):
    imagem = cv2.imread(imagePath)
    scale_percent = 60 # percent of original size
    width = int(imagem.shape[1] * scale_percent / 100)
    height = int(imagem.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    imagem = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    print("Imagem de treino ->" +imagePath)
    caracteristicas = momentosZernike(imagem)
   
    imagePath=imagePath.replace("\\",";") # no windows, a barra é invertida, entao trocar barra por ;
    labels.append(imagePath.split(";")[2])
    x=(caracteristicas)
    dados.append((x))
    # grava dados em arquivo csv
    with open(r'gestosLibras.csv', 'a') as arquivo:
       writer = csv.writer(arquivo)
       writer.writerow(caracteristicas)
    with open(r'nomearquivo.csv', 'a') as arquivo2:
       writer = csv.writer(arquivo2)
       writer.writerow(imagePath.split(";")[2])

print( "[STATUS] Criando classificador...")
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME.R",
                         n_estimators=500)

print( "[STATUS] Ajustando os dados aos modelos..")
bdt.fit(dados, labels)

# codigo que le uma imagem do disco e classifica. 

#loop nas imagens de teste
test_path = "gestos/teste/"
for arquivos in glob.glob(test_path + "*.jpg"):
   
    # le imagem de teste
    arquivos=arquivos.replace(';','//')
    imagem = cv2.imread(arquivos)

    scale_percent = 60 # percent of original size
    width = int(imagem.shape[1] * scale_percent / 100)
    height = int(imagem.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    imagem = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    
    #print(arquivos)
    momentos = momentosZernike(imagem)
    caracteristicas=np.asarray(momentos)
    caracteristicas=(momentos)

   # realiza a predição para a imagem de teste lida
    predicao = bdt.predict(np.asarray(caracteristicas).reshape(1,-1))[0]
    print("Letra ->  ",predicao[0])
    # show the label
    saidaX.append(caracteristicas)
    saidaLabel.append(predicao[0])
    cv2.putText(imagem, predicao[0], (40,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    print( "predicao - {}".format(predicao))
    # display the output image
    cv2.namedWindow('Imagem de Saida',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image',  round(imagem.shape[1]/2), round(imagem.shape[2]/2))
    cv2.imshow("Imagem de Saida", imagem)
    cv2.waitKey(3000)

cv2.destroyAllWindows()





