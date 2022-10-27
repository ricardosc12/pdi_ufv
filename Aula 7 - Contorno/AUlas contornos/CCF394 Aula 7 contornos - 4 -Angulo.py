import cv2
import numpy as np
image = cv2.imread("figuraangulo.png")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray,200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Imagem Binaria', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(image=thresh,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(image=thresh,
#                                       mode=cv2.RETR_EXTERNAL,
#                                       method=cv2.CHAIN_APPROX_NONE)
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
        print(f"bounding_box({x:.2f},{y:.2f},{w:.2f},{h:.2f})")
        img= cv2.rectangle(image_copy,(x, y),(x+w, y+h),(0,255,0),2)
        area = cv2.contourArea(c)
        # Area text
        
        #centroide
        M = cv2.moments(c)
        centx = int(M['m10'] / M['m00'])
        centy = int(M['m01'] / M['m00'])
        cv2.circle(img, (centx,centy), 3, color, thickness)
        (xe,ye),(ma,Ma), angle = cv2.fitEllipse(c)
        print(f"Angulo: {angle:.2f} Maior: {Ma:.2f}  Menor: {ma:.2f}")
        cv2.putText(img, "Angulo: " + str(int(angle)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, 16)
        
        
        

        i=i+1
print("Quantidade de objetos: "+str(len(contours)))
#cv2.imshow('Usando CHAIN_APPROX_NONE', image_copy)
cv2.imshow('Boundinx box', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

