import cv2
import numpy as np
from scipy import ndimage

class HighlightImage:
    def __init__(self, srcImage : str):
        try:
            self.__sourceImg = cv2.imread(srcImage)
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def getSourceImg(self):
        return self.__sourceImg
    
    def editImage(self):
        try:
            img = self.__sourceImg.copy()
            col, lin, _ = img.shape
            img = img[215:col, :lin]

            for j in range(0, col - 215):
                for i in range(0, lin):
                    (blue, green, red) = img[j, i]
                    blueDistance = int(green) - int(blue)
                    redDistance = int(green) - int(red)
                    # Criando contraste maior entre o verde e o resto da imagem
                    if blueDistance > 2 and redDistance > 2:
                        img[j, i, ] = [255, 255, 255]
                    else:
                        img[j, i, ] = [0, 0, 0]
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyMeanFilter(self, srcImg : object, kernelSize : tuple):
        try:
            imgResult = cv2.blur(srcImg, ksize=kernelSize)
            return imgResult
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))

    def applyMedianFilter(self, srcImg : object, kernelSize : int):
        try:
            imgResult = cv2.medianBlur(srcImg, kernelSize)
            return imgResult
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyGaussianFilter(self, srcImg : object, kernelSize : tuple):
        try:
            imgResult = cv2.GaussianBlur(src=srcImg, ksize=kernelSize, sigmaX=0)
            return imgResult
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applySobelFilter(self, srcImg : object, x : int, y : int, kernelSize : int):
        try:
            sobel = cv2.Sobel(src=srcImg, ddepth=cv2.CV_64F, dx=x, dy=y, ksize=kernelSize)
            filteredImage = cv2.convertScaleAbs(sobel)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyCannyFilter(self, srcImg : object, firstTreshold : int, secondTreshold : int):
        try:
            filteredImage = cv2.Canny(srcImg, firstTreshold, secondTreshold)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyRobertsFilter(self, srcImg : object):
        try:
            crossVertical = np.array([
                [1,  0],
                [0, -1]
            ])
            crossHorizontal = np.array([
                [0,  1],
                [-1, 0]
            ])
            copiedImg = srcImg.copy().astype('float64')
            copiedImg /= 255
            verticalArray = ndimage.convolve(copiedImg, crossVertical)
            horizontalArray = ndimage.convolve(copiedImg, crossHorizontal) 
            filteredImage = np.sqrt(np.square(horizontalArray) + np.square(verticalArray))
            filteredImage *= 255
            filteredImage = filteredImage.astype(srcImg.dtype)
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def applyPrewittFilter(self, srcImg : object):
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
            imgPrewittX = cv2.filter2D(srcImg, -1, kernelX)
            imgPrewittY = cv2.filter2D(srcImg, -1, kernelY)
            filteredImage = imgPrewittX + imgPrewittY
            return filteredImage
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))