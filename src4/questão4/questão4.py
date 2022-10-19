from NoiseFilter import NoiseFilter
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reduzir Video
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip("assets/cruzeiro_inter.mp4", 0, 17, targetname="assets/cruzeiro_inter_reduzido.mp4")

# Para otimizar o processo, foi redimensionado e reduzido o tempo do vídeo

filters = NoiseFilter("aviao.jpg", True)
video = cv2.VideoCapture('assets/main_jogo.mp4')
success, video_frame = video.read()
filters.setImage(video_frame)
img_media_3x3 = filters.applyMeanFilter(video_frame, (3, 3))

col, lin, _ = img_media_3x3.shape

# Algoritmo para segmentação de cor, diferente do hsv em que se pegava um range de cor, aqui nós pegamos a distance entre os canais de cores,
# assim podemos pegar diversos tons de vermelho, ou qualquer que seja a cor. Em contrapartida, o algoritmo para segmentação RGB é bem mais demorado que em HSV, já que lá usamos 
# apenas range de H_VALUE
# Além disso geramos uma máscara única contendo os dois times neste método.

def segmentColor(img):
    for j in range(0,col):
        for i in range(0,lin):
            b, g, r = img[j,i]

            # Azul
            b_g_distance = int(b) - int(g)
            b_r_distance = int(b) - int(r)

            # Vermelho
            r_b_distance = int(r) - int(b)
            r_g_distance = int(r) - int(g)

            # Tons de azul ficam totalmente azul
            if b_g_distance>80 and b_r_distance>80 and int(b)>150:
                img[j,i] = [255,0,0]
            # Tons de vermelho ficam totalmente vermelho
            elif r_b_distance>50 and r_g_distance>50 and int(r)>140:
                img[j,i] = [0,0,255]
            else:
                img[j,i] = [0,0,0]


final_video = cv2.VideoWriter("assets/output_main.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (800,380))

while True:
    success, video_frame = video.read()

    if success:

        # Aplica filtro de média
        filters.setImage(video_frame)
        img_media_3x3 = filters.applyMeanFilter(video_frame, (3, 3))

        # Segmenta o frame
        segmentColor(img_media_3x3)

        # Aplica dilatação e erosão
        element = cv2.getStructuringElement(0, (2*10 + 1, 2*10+1), (10, 10))
        dilatation_dst = cv2.dilate(img_media_3x3, element)

        element = cv2.getStructuringElement(2, (2*4 + 1, 2*4+1), (4, 4))
        erosion_dst = cv2.erode(dilatation_dst, element)

        # Sabemos que era pra ser feito e gerado em vídeo um bitwise entre a máscara e o vídeo original, apenas trocamos e fizemos uma
        # adição para visualizarmos melhor o conjunto - time e campo
        # A imagem que mostra o bitwise está sem assets/bitwise.png

        # img2 = cv2.bitwise_and(video_frame, erosion_dst)

        dst = cv2.add(video_frame,erosion_dst)

        final_video.write(dst)
    else:
        print("FIM")
        break

final_video.release()
