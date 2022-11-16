import cv2
import pandas as pd

def min_max_scaling(column):
    return(column-column.min())/(column.max()-column.min())

momentos=[]

imagens=['diamond','quadrado','circulo','triangulo','poligono']
i=len(imagens)
for j in range(len(imagens)):
    image = cv2.imread(imagens[j]+".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=255-image
    cv2.imshow("Figura",image)
    cv2.waitKey(1000)
    cv2.HuMoments(cv2.moments(image)).flatten()
    momentos.append(cv2.HuMoments(cv2.moments(image)).flatten())

df = pd.DataFrame(momentos, columns = ['A','B','C','D','E','F',"G"])

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


















