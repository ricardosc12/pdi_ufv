#Entregar, junstar imagens, video
#Relatório vai ser referente a isso aqui
#Trabalho será sobre juntar imagens

import cv2
import numpy as np

def mostrar_imagem(titulo,img):
    cv2.imshow(titulo, img)
    cv2.waitKey(0)

img = cv2.imread("cores.png")
mostrar_imagem("Original",img)

b,g,r = cv2.split(img)

lin, col, dim = img.shape
img3 = np.zeros([ lin, col*3, ])
img3[0:lin,0:col] = b[0:lin,0:col]
img3[0:lin, col:2*col] = g
img3[0:lin, col:3*col] = r

mostrar_imagem("Bandas",img3)

imagem = cv2.merge([b,g,r])

mostrar_imagem("Bandas unidas",imagem)

#Convertendo pra cinza
gray = cv2.cvtColor(img,cv2.COLOR_BRG2GRAY)
mostrar_imagem("Cinza",gray)

img = cv2.imread("ayrton-senna.jpg")
mostrar_imagem("Original",img)

flipped = cv2.flip(img,1)
mostrar_imagem("Horizontal",flipped)

flipped = cv2.flip(img,-1)
mostrar_imagem("Vertical",flipped)

image = cv2.rotate(img,cv2.cv2.ROTATE_90_CLOCKWISE)
#cv2.ROTATE_180
mostrar_imagem("Rotacionada",image)

lin, col, dim = img.shape
(cX,cY) = (lin//2, col//2)

M = cv2.getRotationMatrix2D((cX,cY),45,1.0)
rotated = cv2.wardAffine(img,M,(lin,col))

mostrar_imagem("Rotated by 45",rotated)

#Pixel a pixel

imagem = cv2.imread("Halteres.jpg")
[col,lin,dim] = image.shape

imagem2=imagem.copy()

print("Tipo da imagem:",image.dtype)
print("Tamanho em pixels da imagem",image.size)
print("linha= ",lin," col= ",col)

for j in range(0,col-1):
    for i in range(0,lin-1):
        (b,g,r) = imagem[j,i]
        if((b>110 and b<255) and (g>80 and g<120) and (r>30 and r<80)):
            b=255
            g=0
            r=0
            imagem2[j,i,] = np.array([b,g,r])
        else:
            imagem2[j,i,] = np.array([0,0,0])

imagem = cv2.hconcat([imagem,imagem2])
mostrar_imagem("Halteres Alterado",imagem)

#VIDEO

camera = cv2.VideoCapture("videocanetas.mp4")
out_file = "saida.avi"

(sucesso, frame) = camera.read()
vid_writter = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M','J','P','G'), 60, 
(frame.shape[1],frame.shape[0]))

numero=1

while True:
    (sucesso, frame) = camera.read()
    if not sucesso:
        break
    cv2.imwrite("Nome"+str(numero)+'.jpg',frame)
    numero+=1

    lab = cv2.cvtColor(frame,cv2.COLOR_RGB2Lab)
    vid_writter.write((lab).astype(np.uint8))

    cv2.imshow("Exibindo video")

    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord("S"):
        break

vid_writter.release()
cv2.destroyAllWindows()