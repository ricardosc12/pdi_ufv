# -*- coding: utf-8 -*-
"""ccf394 svm fashion-mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12Ta3QDXMeWuaBVq3G2565gYrQdy7MxPk

Classificação do Fashon-mnist com svm

material adaptado de https://www.kaggle.com/code/juneyao666/fashion-mnist-classification-machine-learning
"""



import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

traindata = pd.read_csv("fashion-mnist_test.csv")
testdata = pd.read_csv("fashion-mnist_test.csv")

#1.2 Seperate data and label
data_train=traindata.iloc[:,1:785]/ 255.0
label_train=pd.DataFrame([traindata.iloc[:,0]]).T
data_test=testdata.iloc[:,0:784]/ 255.0#1.2 Seperate data and label



categoryMap={0 :'T-shirt/Top',
1 :'Trouser',
2 :'Pullover',
3 :'Dress',
4 :'Coat',
5 :'Sandal',
6 :'Shirt',
7 :'Sneaker',
8 :'Bag',
9 :'Ankle boot'}
label_train['category']=label_train['label'].map(categoryMap)
print(f'Numero de labels: {label_train.value_counts()}')
L = 5
W = 6
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in range(30):
    axes[i].imshow(data_train.values.reshape((data_train.shape[0], 28, 28))[i], cmap=plt.get_cmap('gray'))
    axes[i].set_title("class " + str(label_train['label'][i]) + ": "+ label_train['category'][i])
    axes[i].axis('off')
plt.show()

"""Check for null and missing values"""



#Split training and valdiation set
l_train=pd.DataFrame([traindata.iloc[:,0]]).T
X_train, X_val, Y_train, Y_val = train_test_split(data_train, l_train, test_size = 0.25, random_state=255)


#2.3 Standardizing
np.mean(X_train.values),np.std(X_train.values),np.mean(X_val.values),np.std(X_val.values)

X_train=StandardScaler().fit_transform(X_train)
X_val=StandardScaler().fit_transform(X_val)
np.mean(X_train),np.std(X_train),np.mean(X_val),np.std(X_val)

column_name=['pixel'+str(i) for i in range(1,785)]
X_train = pd.DataFrame(X_train,columns =column_name)
X_val = pd.DataFrame(X_val,columns =column_name)

"""Redução de dimensionalidade com PCA"""

X_train.shape

pca = PCA(n_components=0.9,copy=True, whiten=False)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
print(pca.explained_variance_ratio_)

X_train.shape

"""plotando a Variância Explicada"""

var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
fig = go.Figure(data=go.Scatter(x=list(range(1,len(var)+1)), y=var))
fig.update_layout(title='PCA Variance Explained',
                   xaxis_title='# Of Features',
                   yaxis_title='% Variance Explained')
fig.show()

pcn=X_train.shape[1]
X_train = pd.DataFrame(X_train,columns =column_name[0:pcn])
X_val = pd.DataFrame(X_val,columns =column_name[0:pcn])

"""Criando nosso modelo"""

start_time = time.time()
#svc = svm.SVC(decision_function_shape='ovo')
svc = svm.SVC(kernel="rbf", C = 1.0,gamma="auto")
#A função ravel () é usada para retornar uma matriz 1D contendo todos os elementos da matriz de entrada n-dimensional.
svc.fit(X_train, Y_train.values.ravel())

"""Avaliando o modelo"""

y_train_prd = svc.predict(X_train)
y_val_prd = svc.predict(X_val)
acc_train_svc=accuracy_score(Y_train,y_train_prd )
acc_val_svc=accuracy_score(Y_val,y_val_prd)
print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_svc*100,acc_val_svc*100))
print("--- %s seconds ---" % (time.time() - start_time))

con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Teste' ),pd.Series(y_val_prd, name='Previsto'))
plt.figure(figsize = (9,6))
plt.title("Matriz de Confusão do Classificador SVM")
sns.heatmap(con_matrix, cmap="coolwarm", annot=True, fmt='g')
plt.show()
