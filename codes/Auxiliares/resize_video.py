import cv2
import numpy as np


video = cv2.VideoCapture("original.mp4")
video_resize = cv2.VideoWriter("video_resize.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,288))

while True:
    (sucesso, frame) = video.read()
    
    if not sucesso:
        break
    
    b = cv2.resize(frame,(512,288),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    video_resize.write(b)

    if cv2.waitKey(1) & 0xFF == ord("S"):
        break
    
    # numero+=1
    # FPS
    # sleep(0.01)


video_resize.release()
cv2.destroyAllWindows()