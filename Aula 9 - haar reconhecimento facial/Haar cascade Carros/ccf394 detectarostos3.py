from __future__ import print_function
import cv2 
#https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    rostos=0
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 2)
        faceROI = frame_gray[y:y+h,x:x+w]
        rostos=rostos+1
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        smile=smile_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 2)
        for (x2,y2,w2,h2) in smile:
            center_coordinates = (x + x2 + w2//2, y + y2 + h2//2)
            axesLength = (30, 10) 
            angle = 0
            startAngle = 0
            endAngle = 360
            # Red color in BGR 
            color = (0, 0, 255) 
            # Line thickness of 5 px 
            thickness = 2
   
# Using cv2.ellipse() method 
# Draw a ellipse with red line borders of thickness of 5 px 
            frame = cv2.ellipse(frame, center_coordinates, axesLength,  angle, startAngle, endAngle, color, thickness) 
            
        

    cv2.imshow('Capture - Face detection', frame)
    return rostos

face_cascade_name = 'haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'haarcascade_eye_tree_eyeglasses.xml'
smile_cascade_name= 'haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
smile_cascade = cv2.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not smile_cascade.load(cv2.samples.findFile(smile_cascade_name)):
    print('--(!)Error loading smile cascade')
    exit(0)

img = cv2.imread("sada.jpg")
img=cv2.resize(img,(2*img.shape[1],2*img.shape[0]))

cv2.imshow('img', img)

numerorostos=detectAndDisplay(img)
print("Encontrado %d rostos"%(numerorostos))

