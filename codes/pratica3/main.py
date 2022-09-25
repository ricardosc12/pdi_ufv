from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagem(titulo,img):
    cv2.imshow(titulo, img)
    # cv2.waitKey(0)

img = cv2.imread("assets/plantioCana.jpg")

[col, lin, dim] = img.shape

img = img[215:col, :lin]

original = img.copy()
    
for j in range(0,col-215):
    for i in range(0,lin):
        (b,g,r) = img[j,i]
        blue_distance = int(g) - int(b)
        red_distance = int(g) - int(r)
        # Criando contraste maior entre o verde e o resto da imagem
        if(blue_distance>2 and red_distance>2):  #... or (int(r)==int(g) and int(g)>50 and int(b)<150)
            # boost_green = int(g)*2 if int(g)*2<255 else 255
            # img[j,i,] = [b,boost_green,r]
            img[j,i,] = [255,255,255]
        else:
            img[j,i,] = [0,0,0]

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,0,0)
cv2.imshow("Filtro Canny",edges)
cv2.waitKey(0)

sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) 
filtered_image_x = cv2.convertScaleAbs(sobelx)
 
sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
filtered_image_y = cv2.convertScaleAbs(sobely)
 
sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
filtered_image_xy = cv2.convertScaleAbs(sobelxy)

plt.figure(figsize=(18,19))
plt.subplot(221)
plt.imshow(original[...,::-1])
plt.title('Original') 
plt.axis("off")
 
plt.subplot(222)
plt.imshow(filtered_image_x, cmap='gray')
plt.title('Sobel X') 
plt.axis("off")
 
plt.subplot(223)
plt.imshow(filtered_image_y, cmap='gray')
plt.title('Sobel Y') 
plt.axis("off")
 
plt.subplot(224)
plt.imshow(filtered_image_xy, cmap='gray')
plt.title('Sobel X Y')
plt.axis("off")
plt.show()

# cv2.imshow("Teste", img)
cv2.waitKey(0)
cv2.destroyAllWindows()