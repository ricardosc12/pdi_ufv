import random

import cv2
import numpy as np

def main():
    kernal = np.ones((5, 5), np.uint8)
    image = cv2.imread("figuras2.png", 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

    # dilatando para remover imperfeicoes
    dilated_image = cv2.dilate(thresh, kernal, iterations=2)

    # retornando contornos
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # criando imagem com zeros
    sample = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    for cnt in contours:

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # area do contorno
        area_cnt = cv2.contourArea(cnt)

        # perimetro do contornos
        perimeter_cnt = cv2.arcLength(cnt, True)

        if int(area_cnt) > 16000:# somente contornos maiores serão utilizados
            cv2.drawContours(sample, [cnt], -1, color, -1) # o último parametro -1 indica que a imagem será colorida.

        
    cv2.imshow("Contoured Image", sample)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
