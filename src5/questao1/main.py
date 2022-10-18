import cv2
import numpy as np

# Numero de clusters para o kmeans
try:
    K = int(input("Digite o numero de clusters (2 a 5): "))

    img = cv2.imread('desmatamento.png')
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
        cv2.imwrite("saidas/falsacor_k{}.png".format(K), imgFinal)
        # cv2.imshow('Imagem final', imgFinal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
except Exception as e:
    print(e)