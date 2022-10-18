from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("assets/desmatamento_kmeans.png")

[col, lin, dim] = img.shape

image1 = img[0:col, 0:273]
image2 = img[0:col, 293:lin]

def getArea(img,lin):
    area = 0
    for j in range(0,col):
        for i in range(0,lin):
            (b,g,r) = img[j,i]
            if(int(r)==255):
                area+=1
    return area

cv2.imshow("Antes", image1)
cv2.imshow("Depois", image2)

area1 = getArea(image1,273)
area2 = getArea(image2,lin-293)

print("Área desmatada - Antes: ",area1)
print("Área desmatada - Depois: ",area2)

print("Aumento de: {:.2f}%".format(((area2/area1)-1)*100))

cv2.waitKey(0)