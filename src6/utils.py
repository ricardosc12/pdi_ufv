import cv2
from os import listdir
from os.path import isfile, join

def minMaxScaling(column):
    return(column - column.min()) / (column.max() - column.min())

def getImageFromDataset(letter, file):
    image = cv2.imread("dataset_gestos/{}/{}".format(letter, file), 0)
    return preProcessImage(image)

def preProcessImage(img):
    image = cv2.GaussianBlur(img, (5,5), 0)
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return image

def getImagesByLetter(letter):
    path = "./dataset_gestos/{}".format(letter)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files