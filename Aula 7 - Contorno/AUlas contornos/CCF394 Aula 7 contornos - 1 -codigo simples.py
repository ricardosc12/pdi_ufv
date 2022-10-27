import cv2
import numpy as np
#lendo imagem
image=cv2.imread('shapes.png')
cv2.imshow("original",image)
cv2.waitKey()

# convertendo para cinza
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# encontrando bordas com canny
canny_img=cv2.Canny(img_gray,30,200)
cv2.imshow("Bordas",canny_img)
cv2.waitKey()

# encontrando contornos
# cv2.findContours(image, Retrival mode, approximation mode)
# Retrievel mode especifica quais contornos serão extraidos. RETR_EXTERNAL retorna somente o externo
# approximation mode determina o algoritmo utilizado para extrair o contorno

contours, hierarchy=cv2.findContours(canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


#desenhando TODOS os contornos
# a variavel contours é um vetor... vamos imprimir o tamanho ?
print('Total de contornos encontrados  %d' %(len(contours)))
#o uso do  -1 é para mostrar todos os contornos
cv2.drawContours(image,contours,-1,(0,255,0),2)
cv2.imshow("Contornos",image)
cv2.waitKey()

cv2.destroyAllWindows()
