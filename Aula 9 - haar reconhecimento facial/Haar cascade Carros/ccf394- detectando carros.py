import cv2
import random


def detect_cars(frame):
    # 1.15 - escala da imagem para procura dos carros,
    # 7- vizinhos: quanto maior, mais preciso
    cars = cars_cascade.detectMultiScale(frame, 1.1, 7)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(random.randint(100,255), random.randint(100,255), random.randint(100,255)), thickness=2)
    return frame

def Principal():
    EntradaVideo = cv2.VideoCapture('cars2.mp4')
    while EntradaVideo.isOpened():
        ret, frame = EntradaVideo.read()
        tecla = cv2.waitKey(1)
        if ret:        
            cars_frame = detect_cars(frame)
            cv2.imshow('frame', cars_frame)
            cv2.waitKey(30)
        else:
            break
        if tecla == ord('q'): #q para sair.
            break
    EntradaVideo.release()
    cv2.destroyAllWindows()
#inicio    
cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
Principal()
