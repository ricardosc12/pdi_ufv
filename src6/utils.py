import cv2
from os import listdir
from os.path import isfile, join

PATH = "dataset_gestos"

def minMaxScaling(column):
    return(column - column.min()) / (column.max() - column.min())

def getImageFromDataset(letter, file):
    path = "{}/{}/{}".format(PATH,letter, file)
    image = cv2.imread(path, 0)
    return preProcessImage(image)

def preProcessImage(img):
    image = cv2.GaussianBlur(img, (7,7), 0)
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    # ret3,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # image = cv2.erode(image, None, iterations=1)
    image = 255 - image
    return image

def getImagesByLetter(letter):
    path = "./{}/{}".format(PATH,letter)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files