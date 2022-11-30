import face_recognition
import cv2
import PIL
import numpy as np
from PIL import Image, ImageDraw

image = cv2.imread('serie.jpg')

faces_encoding = {'chris':None, 'drew':None, 'julius':None, 'tonya':None, 'greg':None, 'rochelle':None}

for face in faces_encoding.keys():
    image = face_recognition.load_image_file('./know/{}.png'.format(face))
    faces_encoding[face] = face_recognition.face_encodings(image)[0]

known_face_encodings = list(faces_encoding.values())
known_face_names = list(faces_encoding.keys())



def recognition(src):
    face_locations = face_recognition.face_locations(src)
    face_encodings = face_recognition.face_encodings(src, face_locations)
    pil_image = Image.fromarray(src)
    draw = ImageDraw.Draw(pil_image)
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

    del draw
    pil_image = np.array(pil_image)
    pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2RGB)
    return pil_image

output = cv2.VideoWriter('output/video_recognized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280,720))
video = cv2.VideoCapture('video_chris.mkv')


frames = 0
while True:
    success, frame = video.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = recognition(frame)
        output.write(pil_image)
        print('frame: '+str(frames))
        frames+=1
    else:
        break

output.release()

# Display image
# src_image = cv2.imread('serie.jpg')
# src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

# pil_image = recognition(src_image)


# cv2.imshow("teste",image)
# cv2.waitKey(0)
# pil_image.show()
# Save image
# pil_image.save('identified.jpg')