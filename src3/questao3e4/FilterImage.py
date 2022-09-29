import cv2
import numpy as np
from scipy import ndimage

class FilterImage:
    def __init__(self, srcImage : str, grayScale : bool):
        try:
            self.__sourceImg = cv2.imread(srcImage, 0) if grayScale else cv2.imread(srcImage)
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def getSourceImg(self):
        return self.__sourceImg
    
    def applySobelFilter(self, x : int, y : int, kernelSize : int):
        try:
            sobel = cv2.Sobel(src=self.__sourceImg, ddepth=cv2.CV_64F, dx=x, dy=y, ksize=kernelSize)
            filteredImage = cv2.convertScaleAbs(sobel)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyRobertsFilter(self):
        try:
            crossVertical = np.array([
                [1,  0],
                [0, -1]
            ])
            crossHorizontal = np.array([
                [0,  1],
                [-1, 0]
            ])
            copiedImg = self.__sourceImg.copy().astype('float64')
            copiedImg /= 255
            verticalArray = ndimage.convolve(copiedImg, crossVertical)
            horizontalArray = ndimage.convolve(copiedImg, crossHorizontal) 
            filteredImage = np.sqrt(np.square(horizontalArray) + np.square(verticalArray))
            filteredImage *= 255
            filteredImage = filteredImage.astype(self.__sourceImg.dtype)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyPrewittFilter(self):
        try:
            kernelX = np.array([
                [-1, -1, -1],
                [0,   0,  0],
                [1,   1,  1]
            ])
            kernelY = np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ])
            imgPrewittX = cv2.filter2D(self.__sourceImg, -1, kernelX)
            imgPrewittY = cv2.filter2D(self.__sourceImg, -1, kernelY)
            filteredImage = imgPrewittX + imgPrewittY
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyCannyFilter(self, firstTreshold : int, secondTreshold : int):
        try:
            filteredImage = cv2.Canny(self.__sourceImg, firstTreshold, secondTreshold)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyMeanFilter(self, srcImg : object, kernelSize : tuple):
        try:
            imgResult = cv2.blur(srcImg, ksize=kernelSize)
            return imgResult
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyBitwiseAnd(self, firstImg : object, secondImg : object):
        try:
            imgResult = cv2.bitwise_and(firstImg, secondImg)
            return imgResult
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))

    def generateOutput(self, outputDict : dict):
        try:
            for i, valueDict in outputDict.items():
                for j, value in valueDict.items():
                    cv2.imwrite("saidas/{}/{}.jpg".format(i,j), value)
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
