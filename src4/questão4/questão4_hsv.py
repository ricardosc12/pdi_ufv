# Extract an object in HSV color space based on hue
import numpy as np
from NoiseFilter import NoiseFilter
import cv2
from matplotlib import pyplot as plt

video = cv2.VideoCapture('assets/main_jogo.mp4')
filters = NoiseFilter("aviao.jpg", True)

# azul = 114

# vermelho = 162

# Função que ira pegar um segmento de cor específico no modelo HV
def getSegment(h_value,range,img,hsv):
    lower = (h_value-range, 80, 80)
    upper = (h_value+range, 250, 250)
    mask = cv2.inRange(hsv, lower, upper)
    img2 = cv2.bitwise_and(img, img, mask=mask)
    return img2

# Dilatará a image e depois causará erosão
def distorceImg(segment,dilatation,erosion):
    element = cv2.getStructuringElement(0, (2*dilatation + 1, 2*dilatation+1), (dilatation, dilatation))
    dilatation_dst = cv2.dilate(segment, element)
    element = cv2.getStructuringElement(2, (2*erosion + 1, 2*erosion+1), (erosion, erosion))
    erosion_dst = cv2.erode(dilatation_dst, element)
    return erosion_dst

final_video = cv2.VideoWriter("assets/output_main_hsv.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (800,380))

while True:
    success, video_frame = video.read()

    if success:
        hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)

        # Aplicando filtro de média 3x3
        filters.setImage(video_frame)
        video_frame = filters.applyMeanFilter(video_frame, (3, 3))

        # Segmentação dos dois times - Azul e Vermelho
        segment_blue = getSegment(114,3,video_frame,hsv)
        distorce_blue = distorceImg(segment_blue,10,8)

        segment_red = getSegment(162,3,video_frame,hsv)
        distorce_red = distorceImg(segment_red,10,4)

        # Juntando as duas segmentações
        output = cv2.add(distorce_blue,distorce_red)

        # Aplicando no video original 
        output = cv2.add(output,video_frame)

        final_video.write(output)
    else:
        print("FIM")
        break

final_video.release()
