import cv2
import numpy as np

# Lê a imagem
baseImg = cv2.imread("cena.png")

# Redimensiona a imagem em 60%
percentScale = 60
width = int(baseImg.shape[1] * percentScale / 100)
height = int(baseImg.shape[0] * percentScale / 100)
dim = (width, height)
baseImg = cv2.resize(baseImg, dim, interpolation = cv2.INTER_AREA)
  
# Converte a imagem em BGR para as outras escalas
imageBGR = np.copy(baseImg)
imageHSV = cv2.cvtColor(baseImg,cv2.COLOR_BGR2HSV)
imageLAB = cv2.cvtColor(baseImg,cv2.COLOR_BGR2LAB)

# Exibição dos resultados
cv2.imshow('RGB',imageBGR)
cv2.imshow('HSV',imageHSV)
cv2.imshow('LAB',imageLAB)
cv2.waitKey(0)

# Exibindo as bandas isoladas do espaço RGB
B,G,R = cv2.split(imageBGR)
cv2.imshow('R', R)
cv2.imshow('G', G)
cv2.imshow('B', B)
cv2.waitKey(0)

# Exibindo as bandas isoladas do espaço HSV
H,S,V = cv2.split(imageHSV)
cv2.imshow('H', H)
cv2.imshow('S', S)
cv2.imshow('V', V)
cv2.waitKey(0)

# Exibindo as bandas isoladas do espaço LAB
L,A,B2 = cv2.split(imageLAB)
cv2.imshow('L', L)
cv2.imshow('A', A)
cv2.imshow('B', B2)
cv2.waitKey(0)

cv2.destroyAllWindows()
