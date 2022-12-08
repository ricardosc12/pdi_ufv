import os, glob, cv2

def restoreVideo():
    videoPath = "input.mp4"
    video = cv2.VideoCapture(videoPath)
    dim = (1280, 720)
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, dim)
    i = 0
    while True:
        success, frame = video.read()
        if success:
            newFrame = restoreFrame(frame)
            frameResized = cv2.resize(newFrame, dim)
            output.write(frameResized)
            i += 1
        else:
            break
    output.release()

def restoreFrame(frame):
    # TODO: Tratar frame
    pass

if __name__ == "__main__":
    restoreVideo()
