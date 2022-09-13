from time import sleep
import cv2
import numpy as np

def mostrar_imagem(titulo,img):
    cv2.imshow(titulo, img)
    # cv2.waitKey(0)

# Redimensionando imagem - Resolução Alta
img = cv2.imread("assets/img.png")
img = cv2.resize(img, (960, 720))

[col, lin, dim] = img.shape

# Tamanho em pixels de um frame do video
quadradoX, quadradoY = 512,288

# Posição do video na imagem
posX, posY = 428, 402

img2 = img.copy()

vid_writter = cv2.VideoWriter("assets/thanos_ao_vivo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (960,720))

# Será necessário redimensionar video - Auxiliares/resize_video
camera = cv2.VideoCapture("assets/thanos_resize.mp4")

img_final = img.copy()

while True:
    (sucesso, frame) = camera.read()
    # Caso ao invez de imagem estática, outro video, bastaria 
    # (sucess, frame2) = camera2.read() 
    # E gravar neste frame2
    
    if not sucesso:
        break
    
    for j in range(posY,posY+quadradoY):
        for i in range(posX,posX+quadradoX):
            (b,g,r) = frame[j-posY,i-posX]

            # Subsituição de cores muito verde
            if(not g>=195):
                img_final[j,i] = frame[j-posY,i-posX]
            # Mantém pixel da imagem original
            else:
                img_final[j,i] = img[j,i]
        
    vid_writter.write(img_final)

    if cv2.waitKey(1) & 0xFF == ord("S"):
        break


vid_writter.release()
cv2.destroyAllWindows()