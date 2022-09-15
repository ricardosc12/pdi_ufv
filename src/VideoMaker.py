import cv2
# import numpy as np

class VideoMaker:
    '''
        VideoMaker
        ---------
        Classe responsável por criar um vídeo editado a partir
        de uma imagem de fundo e um vídeo com chroma key.

        Atributos
        ---------
            - __backgroundImg (Mat) : Imagem de fundo no tipo da biblioteca OpenCV
            - __video (Any) : Vídeo com o fundo em chroma key
            - __videoResizedDim (tuple) : Dimensões do vídeo redimensionado

        Parâmetros
        ----------
            - imgPath (str) : Caminho de leitura da imagem
            - videoPath (str) : Caminho de leitura do vídeo
            - newDimImg (tuple) : Dimensões da imagem, parâmetro opcional
    '''
    def __init__(self, imgPath : str, videoPath : str, newDimImg : tuple = None):
        try:
            self.__backgroundImg = cv2.imread(imgPath)
            self.__video = cv2.VideoCapture(videoPath)
            self.__videoResizedDim = (0,0)
            if newDimImg != None:
                self.__backgroundImg = cv2.resize(self.__backgroundImg, newDimImg)
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def resizeVideo(self, newDim : tuple):
        '''
            Redimensiona o vídeo frame a frame em uma nova dimensão.

            Parâmetros
            ----------
                - newDim (tuple): Nova dimensão do vídeo (Coluna, Linha)
        '''
        try:
            output = cv2.VideoWriter('assets/video_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, newDim)
            while True:
                success, frame = self.__video.read()
                if success:
                    frameResized = cv2.resize(frame, newDim)
                    output.write(frameResized)
                else:
                    break
            output.release()
            cv2.destroyAllWindows()
            self.__videoResizedDim = newDim
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def makeVideo1(self, posX : int, posY : int):
        '''
            Cria o vídeo final editado a partir da imagem de fundo e o vídeo em uma posição.

            Parâmetros
            ----------
                - posX (int): Posição do eixo X em que o vídeo ficará na imagem
                - posY (int): Posição do eixo Y em que o vídeo ficará na imagem
        '''
        try:
            resizedVideo = cv2.VideoCapture('assets/video_resized.mp4')
            copiedImg = self.__backgroundImg.copy()
            dimImg = (self.__backgroundImg.shape[1], self.__backgroundImg.shape[0])
            finalVideo = cv2.VideoWriter("assets/output1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, dimImg)
            # Tamanho em pixels de um frame do video
            blockX, blockY = self.__videoResizedDim[0], self.__videoResizedDim[1]
            while True:
                success, frame = resizedVideo.read()
                if success:
                    for j in range(posY, posY + blockY):
                        for i in range(posX, posX + blockX):
                            blue, green, red = frame[j-posY, i-posX]
                            # Subsituição de cores verdes pela distancia das outras
                            blue_distance = int(green) - int(blue)
                            red_distance = int(green) - int(red)
                            if blue_distance > 30 and red_distance > 30:
                                copiedImg[j, i] = self.__backgroundImg[j, i]
                            else: # Mantém pixel da imagem original
                                copiedImg[j, i] = frame[j-posY, i-posX]
                    finalVideo.write(copiedImg)
                    if cv2.waitKey(1) & 0xFF == ord("S"):
                        break
                else:
                    break
            finalVideo.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
    
    def makeVideo2(self, posX : int, posY : int, dimOutputVideo : tuple):
        '''
            Cria o vídeo final editado a partir do vídeo de fundo e a imagem em uma posição.

            Parâmetros
            ----------
                - posX (int): Posição do eixo X em que a imagem ficará no vídeo
                - posY (int): Posição do eixo Y em que a imagem ficará no vídeo
                - dimOutputVideo (tuple): Dimensão do vídeo de saída
        '''
        try:
            finalVideo = cv2.VideoWriter("assets/output2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 40, dimOutputVideo)
            blockX, blockY = self.__backgroundImg.shape[1], self.__backgroundImg.shape[0]
            while True:
                success, frame = self.__video.read()
                if success:
                    for j in range(posY, posY + blockY):
                        for i in range(posX, posX + blockX):
                            frame[j, i] = self.__backgroundImg[j-posY, i-posX]
                    finalVideo.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord("S"):
                        break
                else:
                    break
            finalVideo.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("Ocorreu um erro: {}".format(e))
