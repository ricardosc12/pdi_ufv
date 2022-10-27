import cv2
import argparse
import numpy as np



# CARREGA A IMAGEM E CONVERTE PARA CINZA
img = cv2.imread("shapes.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold = 100 # determina o valor de threshold inicial
rthresh2 = cv2.Canny(gray, threshold, threshold * 2)

# encontra os contornos externos -> RETR_EXTERNAL
contours, _ = cv2.findContours(rthresh2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
clone = img.copy()

# --------------- Centroides ----------------------
#
# loop vetor de contornos
for c in contours:
    # utulizando a função moments, calcula o centroide de cada contorno
    #(Moments retorna os momentos de uma imagem, que são medidas de distribuição de pixels em uma imagem)
    # os momentos m00, m01 e m10 são utilizados para calcular os centroides.
    # https://theailearner.com/tag/cv2-moments/
    moments = cv2.moments(c)  
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    # desenha um circulo no centroide
    cv2.circle(clone, (center_x, center_y), 10, (0, 255, 0), thickness=-1)


cv2.imshow("Centroides", clone)
cv2.waitKey(0)

# --------------- Area e Perimetro --------------

for (i, c) in enumerate(contours):
    # calcule  area and the perimetro
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)
    print("Contorno #{} --\tÁrea: {:.2f}, \t\tPerímetro: {:.2f}".format(i+1, area, perimeter))

    # desenha o contorno
    cv2.drawContours(clone, [c], contourIdx=-1, color=(0, 255, 0), thickness=2)

    # calcula o centro novamente e escreve o numero
    moments = cv2.moments(c)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    cv2.putText(clone, "#{}".format(i+1),
                org=(center_x-20, center_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=4)

# 
cv2.imshow("Contours", clone)
cv2.waitKey(0)

# ---------------- Bounding Box -----------------
#
# copia a imagem original
clone = img.copy()

# 
for c in contours:
    # fencontra o bounding box do objeto
    (x, y, width, height) = cv2.boundingRect(c)
    cv2.rectangle(clone, pt1=(x, y), pt2=(x+width, y+height),
                  color=(0, 255, 0), thickness=2)


cv2.imshow("Bounding Boxes", clone)
cv2.waitKey(0)

# ---------------- Rotated Bounding Box -------------
#
# copai da imagem
clone = img.copy()
# loop over the contours
for c in contours:
    # desenha o bounding box, mesmo se o objeto está rotacionado
    # desta vez, retorna uma tupla... ((x, y), (width, height), angle)
    box = cv2.minAreaRect(c)
    # numpy.int0 is an alias of numpy.intp, which is a type for indexing.
    # It is equivalent to C's ssize_t, either int32 or int64
    box = np.int0(cv2.boxPoints(box))
    cv2.drawContours(clone, [box], contourIdx=-1, color=(0, 255, 0), thickness=2)

# 
cv2.imshow("Rotated Bounding Box", clone)
cv2.waitKey(0)

# ---------------- Minimum Enclosing Circle ------------
#
# 
clone = img.copy()
# loop over the contours
for c in contours:
    # menor circulo contornando a imagem
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(clone, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)

# show the output image
cv2.imshow("Menor circulo que contorna a imagem", clone)
cv2.waitKey(0)

