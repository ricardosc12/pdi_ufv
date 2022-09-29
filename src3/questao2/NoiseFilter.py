import cv2
import numpy as np
import matplotlib.pyplot as plt

class NoiseFilter:
    def __init__(self, srcImage : str, grayScale : bool):
        try:
            self.__sourceImg = cv2.imread(srcImage, 0) if grayScale else cv2.imread(srcImage)
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def getSourceImg(self):
        return self.__sourceImg
    
    def applySaltAndPepperNoise(self, saltAndPepperRate : float, amountOfSaltAndPepper : float):
        try:
            saltAndPepperImg = self.__sourceImg.copy()

            saltAmount = np.ceil(amountOfSaltAndPepper * saltAndPepperImg.size * saltAndPepperRate)
            pepperAmount = np.ceil(amountOfSaltAndPepper * saltAndPepperImg.size * (1 - saltAndPepperRate))

            coordenatesSalt = [np.random.randint(0, i - 1, int(saltAmount)) for i in saltAndPepperImg.shape]
            coordenatesPepper = [np.random.randint(0, i - 1, int(pepperAmount)) for i in saltAndPepperImg.shape]

            saltAndPepperImg[tuple(coordenatesSalt)] = 255
            saltAndPepperImg[tuple(coordenatesPepper)] = 0

            return saltAndPepperImg
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))

    def applyGaussianNoise(self, mean : float):
        try:
            gaussSamples = np.random.normal(mean, 1, self.__sourceImg.shape)
            gauss = gaussSamples.reshape(self.__sourceImg.shape)
            gaussianImg = self.__sourceImg + gauss
            gaussianImg = gaussianImg.astype('uint8')

            return gaussianImg
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
    
    def generateHistogram(self, srcImg : object):
        try:
            histogramImg, _ = np.histogram(srcImg, bins=256, range=(0,256))
            return histogramImg
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))

    def generateOutput(self, outputDict : dict):
        for i, valueDict in outputDict.items():
            for j, value in valueDict.items():
                if "Hist" in j:
                    plt.bar(np.arange(256), value)
                    plt.savefig("saidas/{}/{}.jpg".format(i,j))
                    plt.close()
                else:
                    cv2.imwrite("saidas/{}/{}.jpg".format(i,j), value)
