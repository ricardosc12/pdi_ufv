import cv2 # Biblioteca OpenCV
import pandas as pd # Biblioteca pandas para gerar o csv
from sklearn.preprocessing import MinMaxScaler # Biblioteca para transformar as features em um range
from os import listdir
from os.path import isfile, join

def min_max_scaling(column):
    return(column-column.min())/(column.max()-column.min())

momentos = []
result = []
letras = ['A','B','C','D','E','F','G','I','L','M','N']
count = 0

for letra in letras:
    path = "./dataset_gestos/{}".format(letra)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        image = cv2.imread("dataset_gestos/{}/{}".format(letra, file),0)
        # success, image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        image = cv2.medianBlur(image,5)
        image = cv2.adaptiveThreshold (image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        # image = cv2.bitwise_not(image)
        # success, image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
        # image = 255 - image
        momentos.append(cv2.HuMoments(cv2.moments(image)).flatten())
        result.append(count)

        # cv2.imshow("Imagem em escala de cinza", image)
        # cv2.waitKey(0)
        # break
    count+=1

df = pd.DataFrame(momentos, columns = ['A','B','C','D','E','F','G'])

for col in df.columns:
    df[col] = min_max_scaling(df[col])

scaler = MinMaxScaler()
scaler.fit(df)
scaled = scaler.fit_transform(df)
scaledMoments = pd.DataFrame(scaled, columns=df.columns)
dfFinal = scaledMoments.round(decimals=2)
print(scaledMoments.head(10))
dfFinal['Y'] = result
dfFinal.to_csv("dados.csv", header=False, index=False)
