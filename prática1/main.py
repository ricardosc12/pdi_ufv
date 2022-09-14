from time import sleep
import cv2
import numpy as np

def mostrar_imagem(titulo,img):
    cv2.imshow(titulo, img)
    # cv2.waitKey(0)

# Redimensionando imagem - Resolução Alta
img = cv2.imread("assets/img2.png")
img = cv2.resize(img, (960, 720))

[col, lin, dim] = img.shape

# Tamanho em pixels de um frame do video
quadradoX, quadradoY = 512,288

# Posição do video na imagem
posX, posY = 350, 370

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

            # Subsituição de cores verdes pela distancia das outras
            blue_distance = int(g) - int(b)
            red_distance = int(g) - int(r)

            if(blue_distance>30 and red_distance>30):
                img_final[j,i] = img[j,i]
            # Mantém pixel da imagem original
            else:
                img_final[j,i] = frame[j-posY,i-posX]
                
        
    vid_writter.write(img_final)

    # Verificar
    # cv2.imshow("TEste", img_final)
    # cv2.waitKey(0)
    # break

    if cv2.waitKey(1) & 0xFF == ord("S"):
        break


vid_writter.release()
cv2.destroyAllWindows()