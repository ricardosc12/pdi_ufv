import cv2

def detect_pessoas(frame,escala):
    # 1.15 - escala da imagem para procura dos carros,
    # 7- vizinhos: quanto maior, mais preciso
    pessoas = pessoas_cascade.detectMultiScale(frame, escala, 5)
    for (x, y, w, h) in pessoas:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def Principal():
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

    EntradaVideo = cv2.VideoCapture('walking.avi')
    while EntradaVideo.isOpened():
        ret, frame = EntradaVideo.read()
        tecla = cv2.waitKey(1)
        if ret:        
            pessoas_frame = detect_pessoas(frame,1.1)
            pessoas_frame = detect_pessoas(pessoas_frame,2)
            
            cv2.imshow('frame', pessoas_frame)
            out.write(pessoas_frame)  
            cv2.waitKey(1)
        else:
            break
        if tecla == ord('q'):
            break

    EntradaVideo.release()
    cv2.destroyAllWindows()
    
pessoas_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
Principal()
