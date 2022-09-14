import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def showImage(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)

# Lê a webcam e pega o primeiro frame
# webCam = cv2.VideoCapture(0)
# (success, frame) = webCam.read()
# cv2.imwrite("sources/myphoto.jpg", frame)
# cv2.destroyAllWindows()

# Lê a imagem gravada
backgroundImg = cv2.imread("sources/myphoto.jpg")
col, lin, dim = backgroundImg.shape
print("Dimensoes da imagem: {}x{}x{}".format(col, lin, dim))
showImage("Hello", backgroundImg)

# Lê o vídeo e corta a parte que começa com o fundo verde
ffmpeg_extract_subclip("sources/spinner.mp4", 11, 72, targetname="sources/spinner_cutted.mp4")

# Lê o vídeo cortado e redimensiona frame a frame para o mesmo tamanho da imagem
video = cv2.VideoCapture("sources/spinner_cutted.mp4")
newLin, newCol = 360, 240
output = cv2.VideoWriter('sources/spinner_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (newLin, newCol))
while True:
    success, frame = video.read()
    if success:
        frameResized = cv2.resize(frame, (newLin, newCol))
        output.write(frameResized)
    else:
        break
output.release()
cv2.destroyAllWindows()

# Sobrepõe vídeo na imagem
resizedVideo = cv2.VideoCapture("sources/spinner_resized.mp4")
copiedImg = backgroundImg.copy()
finalVideo = cv2.VideoWriter("sources/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (lin, col))
# Tamanho em pixels de um frame do video
blockX, blockY = newLin, newCol
# Posição do video na imagem
posX, posY = 0, 100
while True:
    success, frame = resizedVideo.read()
    croppedFrame = frame[:,50:]
    if success:
        for j in range(posY, posY + blockY):
            for i in range(posX, posX + blockX-50):
                blue, green, red = croppedFrame[j - posY, i - posX]
                if green >= 195 and blue < 140 and red < 140: # Subsituição de cores muito verde
                    copiedImg[j, i] = backgroundImg[j, i]
                else: # Mantém pixel da imagem original
                    copiedImg[j, i] = croppedFrame[j - posY, i - posX]
        finalVideo.write(copiedImg)
        if cv2.waitKey(1) & 0xFF == ord("S"):
            break
    else:
        break
finalVideo.release()
cv2.destroyAllWindows()