from NoiseFilter import NoiseFilter
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Reduzir Video
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip("assets/cruzeiro_inter.mp4", 0, 17, targetname="assets/cruzeiro_inter_reduzido.mp4")

filters = NoiseFilter("aviao.jpg", True)
video = cv2.VideoCapture('assets/cruzeiro_inter_reduzido.mp4')
success, video_frame = video.read()
col, lin, _ = video_frame.shape

final_video = cv2.VideoWriter("assets/main_jogo.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 35, (800,380))

while True:
    success, video_frame = video.read()
    if success:
        video_frame = video_frame[140:col,0:lin]
        video_frame = video_frame[0:380,0:lin]
        final_video.write(video_frame)
    else:
        print("FIM")
        break

final_video.release()