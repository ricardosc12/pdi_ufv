from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import numpy as np

# Corte do vídeo

# with VideoFileClip("assets/news_original.mp4") as video:
#     new = video.subclip(45, 55)
#     new.write_videofile("assets/news_clib.mp4", audio_codec='aac')


# Carrega imagem e redimensiona
img = cv2.imread("assets/nana.jpg")
img = cv2.resize(img, (532,280))

# Define espaço para colocar imagem, e posição no video
posX, posY = 140, 163
quadradoX, quadradoY = 532,280

# Saída do vídeo gerado
vid_writter = cv2.VideoWriter("assets/aluno_dorminhoco.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 40, (1280,720))

# Carrega vídeo de entrada
camera = cv2.VideoCapture("assets/news_clib.mp4")

while True:
    (sucesso, frame) = camera.read()
    
    if not sucesso:
        break
    
    # Substitui espaço do vídeo pela imagem
    for j in range(posY,posY+quadradoY):
        for i in range(posX,posX+quadradoX):
            frame[j,i] = img[j-posY,i-posX]
        
    vid_writter.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("S"):
        break



vid_writter.release()
cv2.destroyAllWindows()


# editor = VideoMaker('', '')
# editor.resizeVideo((0,0))
# editor.makeVideo(0,0)