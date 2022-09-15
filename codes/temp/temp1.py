from time import sleep
import cv2
import numpy as np

def mostrar_imagem(titulo,img):
    cv2.imshow(titulo, img)
    # cv2.waitKey(0)

# camera = cv2.VideoCapture("videocanetas.mp4")

img = cv2.imread("img.png")
img = cv2.resize(img, (960, 720))
# mostrar_imagem("Original",img)

[col, lin, dim] = img.shape

# Desenhar quadrado no meio da tela

# centerX, centerY = int(960/2), int(720/2)
quadradoX, quadradoY = 512,288
# drawX, drawY = centerX-int(quadradoX/2), centerY-int(quadradoY/2)
# drawXF, drawYF = drawX+quadradoX, drawY+quadradoY

# if(drawXF>=lin or drawYF>=col):
#     print("error!, limit of image")
#     exit(0)

img2 = img.copy()

#CENTER
# for j in range(drawY,drawYF):
#     for i in range(drawX,drawXF):
#         img2[j,i,] = np.array([0,0,0])
        # (b,g,r) = img[j,i]

# b,g,r = cv2.split(img)

# thanos = cv2.imread("thanos.jpg")
# [col,lin,dim] = thanos.shape
# thanos = cv2.resize(thanos, (int(lin*0.4),int(col*0.4)))

# mostrar_imagem("Thanos",thanos)
# print(thanos.shape)

# img_final = img.copy()

# for j in range(402,402+quadradoY):
#     for i in range(428,428+quadradoX):
#         (b,g,r) = thanos[j-402,i-428]
#         if(not g>=190):
#             img_final[j,i] = thanos[j-402,i-428]
        # img2[j,i,] = np.array([0,0,0])
        # (b,g,r) = img[j,i]


# mostrar_imagem("Draw",img_final)


vid_writter = cv2.VideoWriter("thanos_ao_vivo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (960,720))

# thanos2 = vid_writter = cv2.VideoWriter("thanos_resize.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,288))

camera = cv2.VideoCapture("thanos_resize.mp4")


img_final = img.copy()
numero=1

while True:
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
    
    # b = cv2.resize(frame,(512,288),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    # thanos2.write(b)

    for j in range(402,402+quadradoY):
        for i in range(428,428+quadradoX):
            (b,g,r) = frame[j-402,i-428]
            if(not g>=190):
                img_final[j,i] = frame[j-402,i-428]
            else:
                img_final[j,i] = img[j,i]
        

    vid_writter.write(img_final)


    if cv2.waitKey(1) & 0xFF == ord("S"):
        break
    
    # numero+=1
    # FPS
    # sleep(0.01)


vid_writter.release()
cv2.destroyAllWindows()

# cv2.waitKey(0)