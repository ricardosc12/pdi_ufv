import cv2
import numpy as np
image = cv2.imread("figuras2.png")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Imagem Binaria', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(image=thresh,
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(image=thresh,
#                                       mode=cv2.RETR_EXTERNAL,
#
#se quiser plotar somente os 3 maiores
#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

if len(contours) == 0:
  print( "nenhum contorno encontrado")
else:
  image_copy = image.copy()
  cv2.drawContours(image=image_copy, contours=contours,
                   contourIdx=-1, color=(0, 255, 0),
                   thickness=2, lineType=cv2.LINE_AA)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 0.6
  color = (0, 0, 255)
  thickness = 1
  i=0;
  for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        texto= "Contorno: " +str(i)+" "+np.array2string(hierarchy[0][i])
        cv2.putText(image_copy, texto, (x,y+10), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        i=i+1
print("Quantidade de objetos: "+str(len(contours)))
cv2.imshow('Usando CHAIN_APPROX_NONE', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

