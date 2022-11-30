import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from utils import concat_tile_resize

# Extraindo rostos com haar cascade

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

path = "assets/"
face_count = 0

for file in [f for f in listdir(path) if isfile(join(path, f))]:

    image = cv2.imread(path+file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_gray,(x,y),(x+w,y+h),(255,0,0),2)
        face = image[y:y+h, x:x+w]
        cv2.imwrite('dataset/face_'+str(face_count)+'.png',face)
        face_count+=1

cv2.destroyAllWindows()


# Concatena imagens do elenco


# faces = {'chris':[], 'rochelle':[],'tonya':[],'drew':[],'julius':[],'greg':[]}

# path = 'know/'

# for file in [f for f in listdir(path) if isfile(join(path, f))]:
#     for face in faces.keys():
#         if(face in file):
#             image = cv2.imread(path+file)
#             image = cv2.resize(image, (133,133),interpolation=cv2.INTER_AREA)
#             faces[face].append(image)

# dataset = []
# for face in faces.keys():
#     dataset.append(faces[face])

# im_tile_resize = concat_tile_resize(dataset)

# cv2.imwrite('elenco.png',im_tile_resize)

