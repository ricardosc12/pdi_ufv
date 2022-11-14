import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

kernelType = input("Digite o tipo do kernel (linear ou rbf): ")
df = pd.read_csv('dados.csv')
# Selecionado os dados
X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
Y = df.iloc[:, [7]].values
# Dividindo o pacote de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# Criando o modelo do classificador e obtendo as saidas para as entradas de teste
classificador = svm.SVC(kernel=kernelType, C=0.01)
y_pred = classificador.fit(X_train, y_train).predict(X_test)

# Convertendo para array, para aplicar na matriz de confusão
y_test = np.array(y_test)
y_pred = np.array(y_pred)
np.set_printoptions(precision=2)

# Obtendo a precisao do modelo por meio da metrica de acuracia
precisao = 100*accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {precisao:.3f} %')

newImage = cv2.imread("56.png", 0)
# image = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
# image=255-image
image = cv2.bitwise_not(newImage)
success, image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
if success:
    huMoments = cv2.HuMoments(cv2.moments(image)).flatten()
    saida = classificador.predict([huMoments])
    print(saida)
# rows, cols, cor = newImage.shape
# for y in range(0, rows):
#     for x in range(0, cols):
#         entrada = newImage[y, x]
#         saida = classificador.predict(entrada.reshape(1, -1))
#         if saida == 0:
#             newImage[y, x] = (0, 0, 255)
#         if saida == 1:
#             newImage[y, x] = (255, 0, 0)
#         if saida == 2:
#             newImage[y, x] = (0, 255, 0)
#         if saida == 3:
#             newImage[y, x] = (255, 255, 255)
#         if saida == 4:
#             newImage[y, x] = (100, 100, 100)
# cv2.imwrite("output_{}.png".format(kernelType), newImage)
# cv2.imshow("Imagem resultante", newImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()