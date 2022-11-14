import cv2 # Biblioteca OpenCV
import pandas as pd # Biblioteca pandas para gerar o csv
from sklearn.preprocessing import MinMaxScaler # Biblioteca para transformar as features em um range

def min_max_scaling(column):
    return(column-column.min())/(column.max()-column.min())

momentos = []
result = []
letras = ['A','B','C','D','E','F','G','I','L','M','N']
count = 0
for letra in letras:
    for i in range(2269):
        try:
            image = cv2.imread("dataset_gestos/{}/{}.png".format(letra, i+1), 0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image=255-image
            image = cv2.bitwise_not(image)
            success, image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
            if success:
                momentos.append(cv2.HuMoments(cv2.moments(image)).flatten())
                result.append(count)
        except:
            break
    count += 1

df = pd.DataFrame(momentos, columns = ['A','B','C','D','E','F','G'])
df['Y'] = result
df.to_csv("dados.csv", header=False, index=False)

# for col in df.columns:
#     df[col] = min_max_scaling(df[col])

# scaler = MinMaxScaler()
# scaler.fit(df)
# scaled = scaler.fit_transform(df)
# scaledMoments = pd.DataFrame(scaled, columns=df.columns)
# dfFinal = scaledMoments.round(decimals=2)
# print(scaledMoments.head(10))
# dfFinal['Y'] = result
# dfFinal.to_csv("dados.csv", header=False, index=False)
