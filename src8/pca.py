#https://gogul09.github.io/software/texture-recognition
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from sklearn import svm
from math import copysign, log10
import pandas as pd
from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, graycoprops
import skimage
from sklearn.neural_network import MLPClassifier
from skimage import io, color, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
          # xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def normaliza(a):
        a=np.array(a)
        rowmin=a.min(axis=0)
        rowmax=a.max(axis=0)
        r=rowmax-rowmin
        l,w=a.shape
        new_matrix = np.zeros((l,w))
        row_sums = a.sum(axis=1)
        for i, (row, row_sum) in enumerate(zip(a, row_sums)):
            print(i,row)
            new_matrix[i,:] = (row - rowmax)/(rowmax-rowmin)
        return new_matrix,rowmin,rowmax

def extract_features_mt(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


# function to extract haralick textures from an image
def extract_features(image):
        # calcula descritores de halarick para  4 adjacencias
        image = img_as_ubyte(image)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)

        max_value = inds.max()+1
        glcm =  skimage.feature.graycomatrix(inds, [10], [0], max_value, symmetric=False, normed=True)
        contrast= skimage.feature.graycoprops(glcm, 'contrast')[0][0]
        energy= skimage.feature.graycoprops(glcm, 'energy')[0][0]
        homogeneity= skimage.feature.graycoprops(glcm, 'homogeneity')[0][0]
        correlation=skimage.feature.graycoprops(glcm, 'correlation')[0][0]
        dissimilarity=skimage.feature.graycoprops(glcm, 'dissimilarity')[0][0]
        
        textures = ([contrast,energy,homogeneity,correlation,dissimilarity])
        #textures = ([contrast,homogeneity,correlation])
        
      
        return textures

def lerimagem(file):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        w,h= gray.shape
        #ajustando imagens para 300 pixels de largura
        escala=(300/w)
        width = int(gray.shape[1] * escala )
        height = int(gray.shape[0] * escala )
        dsize = (width, height)
        output = cv2.resize(gray, dsize)
        #cv2.imshow("Imagens Lidas", output)
        #cv2.waitKey(100)
        return output
                
# conjunto de treino
train_path  = "dataset/train/"
train_names = os.listdir(train_path)
# vetores de caracteristicas
train_features = []
train_labels   = []

# extraindo caracteristicas do conjunto de treinamento
# usa as 3 primeiras letras do arquivo como label de saida
print ("[STATUS] Iniciando extração das caracteristicas..")
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name[0]
        i = 1
        #print(cur_path)
        for file in glob.glob(cur_path):
                if (file!="dataset/train/desktop.ini"):  # o  googledrive armazena um arquivo desktop.ini na pasta....
                        #print( " Imagem - {} no {}".format(i, cur_label))
                        gray=lerimagem(file)
                        # extrai descritores de haralick 
                        features = extract_features(gray)
                        #cria vetores de caracteristicas (x) e saida (y)
                        train_features.append(features)
                        train_labels.append(cur_label)
                        i += 1


# Tamanho dos vetores
print ("Treinamento - entrada: {}".format(np.array(train_features).shape))
print ("Treinamento - labels de saida: {}".format(np.array(train_labels).shape))

'''
#Principal Component Analysis é uma metodologia que pode ser utilizada para reduzir o vetor de caracteristicas
# de um espaço n-dimensional, para um m-dimensional, sendo que m<n
# ainda aumenta a variãncia dos dados, permitindo melhor separacao entre eles.
#abaixo, o vetor de caracteristicas (features) é reduzido para n=4
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(train_features)
# criando o classificador
'''

print ("[STATUS] Criando o classificador..")


pca = PCA(n_components=5,copy=True, whiten=False)
pca.fit(train_features)
train_features = pca.transform(train_features)
print("Variancia ",pca.explained_variance_ratio_)
classificador=svm.SVC(kernel='rbf', max_iter=-1, C=20, gamma=20)


#classificador = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
#                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)



#classificador = svm.NuSVC(gamma='auto')
# ajustando o classificador


train_features,minimo, maximo=normaliza(train_features)
print ("[STATUS]Criando o modelo do classificador a partir do par entrada/saida..")

classificador.fit(train_features, train_labels)
test_features = []
test_labels   = []
#iniciando testes
test_path = ("dataset/test")
test_names = os.listdir(train_path)
acertos=0
erros=0
arquivos=[]
for file in glob.glob(test_path + "/*.png"):
        _,nomearquivo=file.split("\\")
        cur_label = nomearquivo[0]
        # lendo imagem de teste
        gray = lerimagem(file)
        arquivos.append(file)
        #print(file)
        #extrai vetor de caracteristicas
        features = extract_features(gray)
        f=np.array(features)
        features = pca.transform(f.reshape(1,-1))
        test_features.append(features)
        test_labels.append(cur_label)

# Fit and transform the data to the model
y_test=[]
y_pred=[]
for i in range(0,len(test_features)):
        features = (test_features[i]-maximo)/(maximo-minimo)
        #l=np.array(test_features[i])
        previsao = classificador.predict(features.reshape(1, -1))
        previsao=previsao[0]
        cur_label=test_labels[i]
        y_test.append(cur_label)
        y_pred.append(previsao)
        if (cur_label==previsao):
                acertos=acertos+1
        else:
                erros=erros+1
        # show the label
        image=cv2.imread(arquivos[i])
        cv2.putText(image, previsao, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        print ("Lido {} - Previsto - {}".format(cur_label,previsao))

        # display the output image
        #cv2.imshow("Resultado", image)
        #cv2.waitKey(100)
print("Acertos: {}, Erros: {}".format(acertos,erros))
from sklearn.metrics import accuracy_score
precisao=100*accuracy_score(y_test,y_pred)
print(f'Precisão do modelo: {precisao:.3f} %')


# exit(0)
image = lerimagem("itai-sp.png")
rows, cols = image.shape


#plot_confusion_matrix(y_test, y_pred, classes=y_test, title='Confusion matrix, without normalization')
#df_train = pd.DataFrame(train_features, columns = ['contrast','energy','homogeneity','correlation','dissimilarity'])

#df_test = pd.DataFrame(test_features, columns = ['contrast','energy','homogeneity','correlation','dissimilarity'])
#df_train.to_csv('train_features.csv')
#df_test.to_csv('test_features.csv')

con_matrix = pd.crosstab(pd.Series(y_test, name='Teste' ),pd.Series(y_pred, name='Previsto'))
plt.figure(figsize = (9,6))
plt.title("Matriz de Confusão do Classificador SVM")
sns.heatmap(con_matrix, cmap="coolwarm", annot=True, fmt='g')
plt.show()



classe = [0]
count = [0]

ipca = PCA(5)

def on_mouse(event, x, y, flags, param):
    if(event == cv2.EVENT_RBUTTONDOWN):
        classe[0]+=1
        count[0]=0
        print(classe[0])
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            entrada = image[y-10:y+9, x-10:x+9]

            cv2.imshow("entrada", entrada)
            # cv2.waitKey(0)

            features = extract_features(entrada)
            f=np.array(features)
            features = pca.transform(f.reshape(1,-1))

            saida = classificador.predict(features.reshape(1, -1))
            print(saida)
        except Exception as e:
            print(e)
            pass


        # entrada = image[y, x]
        # saida = classificador.predict(entrada.reshape(1, -1))
        # if saida == 0:
        #     image[y, x] = (0, 51, 0)
        # if saida == 1:
        #     image[y, x] = (120, 120, 120)
        # if saida == 2:
        #     image[y, x] = (140, 179, 217)
        # if saida == 3:
        #     image[y, x] = (121, 210, 121)
        # if saida == 4:
        #     image[y, x] = (102, 153, 255)

cv2.namedWindow("teste")
cv2.setMouseCallback("teste", on_mouse)
cv2.imshow("teste", image)
cv2.waitKey(0)

# while True:
#     # Mostra a imagem e espera a tecla ser pressionada
#     cv2.imshow("teste", image)
#     key = cv2.waitKey(1) & 0xFF
#     # Se a tecla 'c' é pressionada sai do loop