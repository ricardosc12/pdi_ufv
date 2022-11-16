import cv2, sys, os
import numpy as np
import pandas as pd
from math import copysign, log10
momentos=[];momentosoriginais=[]; mn=[]; centros=[]; area=[]
nomes=("k0.png","k1.png","k2.png","s0.png","s1.png","s2.png","s3.png","s4.png")
print("imprimindo os 7 momentos de hu das imagens")
for i in range(0,len(nomes)):
    filename=nomes[i]
    #print(filename)
    imagemcolor=cv2.imread(filename)
    imagem = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(imagem, 128, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    moment = cv2.moments(im)
    momentosoriginais.append(moment)
    x=(int(moment['m10']/moment['m00']))
    y=(int(moment['m01']/moment['m00']))
    area.append(moment['m00'])
    centros.append([x,y])
    #print(moment)
    huMoments = cv2.HuMoments(moment)
    momentos.append(huMoments)
    print("{}: ".format(filename),end='')
    m=[]

    for i in range(0,7):
        # NORMALIZANDO COM TRASNFORMAÇÃO LOGARITMICA
        print("{:.4f}".format(-1*copysign(1.0,huMoments[i])*log10(abs(huMoments[i]))),end=' ')
        temp=(-1*copysign(1.0,huMoments[i])*log10(abs(huMoments[i])))
    print()
    mn.append(m)
    cv2.drawContours(imagemcolor, cnts, 0, (0, 255, 0), 2)
    cv2.circle(imagemcolor, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(imagemcolor, "centroid", (x - 25, y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow("Image", imagemcolor)
    cv2.waitKey(1000)


'''

    Connected Components
• Method 2: Find contours of connected blobs
 contours, hierarchy = cv2.findcontours(img, mode, method)
 mode – contour retrieval mode
 cv2.RETR_EXTERNAL – retrieves only the extreme outer contours
 cv2.RETR_LIST – retrieves all of the contours

 method – contour approximation method
 cv2.CHAIN_APPROX_NONE – stores all the contour points
 cv2.CHAIN_APPROX_SIMPLE – stores only the corner points
Example:
contours, _ = cv2.findContours(myimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  '''  
