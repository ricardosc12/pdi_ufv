import cv2
import numpy as np

class PrewittWebcam:
    '''
        PrewittWebcam
        -------------
        Classe responsável por criar uma imagem com o filtro
        Prewitt a partir da imagem retirada da webcam.

        Atributos
        ---------
            - __kernelX (NDArray) : Kernel X do filtro
            - __kernelY (NDArray) : Kernel Y do filtro
    '''
    def __init__(self):
        self.__kernelX = np.array([
            [-1, -1, -1],
            [0,   0,  0],
            [1,   1,  1]
        ])
        self.__kernelY = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ])
    
    def filterWebcamImage(self):
        '''
            Aplica o filtro Prewitt na imagem retirada da webcam.
        '''
        webcam = cv2.VideoCapture(0)
        success, img = webcam.read()
        if success:
            imgPrewittX = cv2.filter2D(img, -1, self.__kernelX)
            imgPrewittY = cv2.filter2D(img, -1, self.__kernelY)
            imgFinal = imgPrewittX + imgPrewittY
            cv2.imshow("Imagem filtrada", imgFinal)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Ocorreu um erro na leitura da imagem!")