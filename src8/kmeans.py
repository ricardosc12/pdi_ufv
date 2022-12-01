import cv2
import numpy as np

# Numero de clusters para o kmeans
try:
    K = 5

    img = cv2.imread('itai-sp.png')
    Z = img.reshape((-1,3))
    # Necessario converter para float32
    Z = np.float32(Z)

    # Define os criterios
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Executa o kmeans
    result, label, centro = cv2.kmeans(Z, K, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Extrai os resultados convertendo de volta para uint8 e cria a imagem de saida
    if result:
        centro = np.uint8(centro)
        imgResult = centro[label.flatten()]
        imgResult = imgResult.reshape((img.shape))
        imgFinal = np.concatenate((img, imgResult), axis=1)
        cv2.imwrite("kmeans.png", imgFinal)
except Exception as e:
    print(e)