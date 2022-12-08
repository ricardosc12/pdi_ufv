import os, glob, cv2
import numpy as np

def restoreVideo():
    count = 0
    videoPath = "restoring.mov"
    video = cv2.VideoCapture(videoPath)
    dim = (701,394)
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, dim)
    i = 0
    while True:
        success, frame = video.read()
        if success:
            newFrame = restoreFrame(frame)
            # cv2.imwrite(f"frames/frame{count}.png",frame)
            # count+=1
            # frameResized = cv2.resize(newFrame, dim)
            output.write(newFrame)
            i += 1
        else:
            break
    output.release()

def restoreFrame(frame):
    mask = maskFrame(frame)
    resized = cv2.resize(frame,(701,394))
    return cv2.inpaint(resized, mask, 30, flags=cv2.INPAINT_NS)

def maskFrame(frame):
    img = frame.copy()
    img = cv2.resize(img,(701,394))
    col, lin, _ = img.shape

    for j in range(0, col):
        for i in range(0, lin):
            (blue, green, red) = img[j, i]
            blueDistance = int(green) - int(blue)
            redDistance = int(green) - int(red)
            # Criando contraste maior entre o verde e o resto da imagem
            if blueDistance > 25 and redDistance > 25:
                img[j, i, ] = [255, 255, 255]
            else:
                img[j, i, ] = [0, 0, 0]
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

if __name__ == "__main__":
    restoreVideo()  
    # img = cv2.imread("frame124.png")
    # out = maskFrame(img)

    # img = cv2.resize(img,(701,394))

    # flags = cv2.INPAINT_NS
    # output = cv2.inpaint(img, out, 30, flags=flags)

    # final = np.hstack([img,output])
    # cv2.imshow("out",final)
    # cv2.waitKey(0)
    # 
