import cv2
import numpy as np
import os
import mahotas
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from imutils import paths
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA



def calcula_momentosZernike(image):

    #aplicando filtro de media 
    filtroMedia = cv2.GaussianBlur(image,(5,5),0)
 
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



def min_max_scaling(column):
    return(column-column.min())/(column.max()-column.min())



# programa principal
momentos=[]
momentosPCA=[]

imagens=['diamond','quadrado','circulo','triangulo','poligono']
i=len(imagens)
for j in range(len(imagens)):
    image = cv2.imread(imagens[j]+".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=255-image
    #cv2.imshow("Figura",image)
    cv2.waitKey(1000)
    momentosZernike= calcula_momentosZernike(image)
    print(np.shape(momentosZernike))
    # Create a PCA that will retain 99% of the variance
   
    momentos.append(momentosZernike)


   
colunas=[]

pca = PCA(n_components=0.90, whiten=True)
    # Conduct PCA
momentos=np.array(momentos)
momentos = pca.fit_transform(momentos)
print('Numero de caracteristicas reduzidas para :', momentos.shape[1])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

for j in range(np.shape(momentos)[1]):
               colunas.append(str(j))
df = pd.DataFrame(momentos, columns = colunas)

for col in df.columns:
    df[col]=min_max_scaling(df[col])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
scaled= scaler.fit_transform(df)
scaled_momentos=pd.DataFrame(scaled, columns=df.columns)
t=scaled_momentos.round(decimals=2)
print(imagens)
print(scaled_momentos.head(10))


print(t.head())









