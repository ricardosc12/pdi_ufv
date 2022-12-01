import cv2
import os
import glob
import skimage
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


def extract_features(image):
    image = skimage.img_as_ubyte(image)
    image = np.asarray(image, dtype="int32")

    glcm = skimage.feature.graycomatrix(image, [1], [0], levels=image.max()+1, symmetric=False, normed=True)
    contrast = skimage.feature.graycoprops(glcm, 'contrast')[0][0]
    energy = skimage.feature.graycoprops(glcm, 'energy')[0][0]
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0][0]
    correlation = skimage.feature.graycoprops(glcm, 'correlation')[0][0]
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0][0]

    textures = ([contrast, energy, homogeneity, correlation, dissimilarity])

    return textures


def lerimagem(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# conjunto de treino
train_path = "dataset"
train_names = os.listdir(train_path)
# vetores de caracteristicas
X = []
Y = []

# extraindo caracteristicas do conjunto de treinamento
print("[STATUS] Iniciando extração das caracteristicas..")
for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name[0]
    for file in glob.glob(cur_path):
        gray = lerimagem(file)
        # extrai descritores de haralick
        features = extract_features(gray)
        # cria vetores de caracteristicas (x) e saida (y)
        X.append(features)
        Y.append(cur_label)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0) 

# Tamanho dos vetores
print("Treinamento - entrada: {}".format(np.array(X_train).shape))
print("Treinamento - labels de saida: {}".format(np.array(y_train).shape))


print("[STATUS] Criando o classificador..")
pca = PCA(n_components=5, copy=True, whiten=False)
pca.fit(X_train)
X_train = pca.transform(X_train)
print("Variancia ", pca.explained_variance_ratio_)
classificador = svm.SVC(kernel='linear')


print("[STATUS] Criando o modelo do classificador a partir do par entrada/saida..")
X_test = pca.transform(X_test)
y_pred = classificador.fit(X_train, y_train).predict(X_test)
# Convertendo para array, para aplicar na matriz de confusão
y_test = np.array(y_test)
y_pred = np.array(y_pred)
np.set_printoptions(precision=2)

# Obtendo a precisao do modelo por meio da metrica de acuracia
precisao = 100*accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {precisao:.3f} %')

image = lerimagem("itai-sp.png")
newImage = cv2.imread("itai-sp.png")
rows, cols = image.shape
classe = [0]
count = [0]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            entrada = image[y-10:y+9, x-10:x+9]

            features = extract_features(entrada)
            f = np.array(features)
            features = pca.transform(f.reshape(1, -1))

            saida = classificador.predict(features)
            saida = int(saida[0])
            if saida == 0: # Verde escuro: mata
                newImage[y-10:y+9, x-10:x+9] = (0, 51, 0)
            if saida == 1: # Cinza: Cidade
                newImage[y-10:y+9, x-10:x+9] = (120, 120, 120)
            if saida == 2: # Laranja: Desmatamento
                newImage[y-10:y+9, x-10:x+9] = (140, 179, 217)
            if saida == 3: # Verde claro: Plantio
                newImage[y-10:y+9, x-10:x+9] = (121, 210, 121)
            if saida == 4: # Marrom: Terra
                newImage[y-10:y+9, x-10:x+9] = (102, 153, 255)
            cv2.imshow("Resultado", newImage)
        except Exception as e:
            print(e)
            pass


cv2.namedWindow("Selecione um pixel")
cv2.setMouseCallback("Selecione um pixel", on_mouse)
cv2.imshow("Selecione um pixel", image)
cv2.waitKey(0)
