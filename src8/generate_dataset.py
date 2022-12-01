import cv2
import numpy as np

classe = [0]
count = [0]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        image = img[y-10:y+9, x-10:x+9]
        cv2.imwrite(f'dataset/{classe[0]}_{count[0]}.png', image)
        count[0] += 1
        print(f"Imagem num: {count[0]}")

        
arquivo = "dados.csv"
numClasses = 5

for j in range(0, numClasses):
    print("Coletando para o agrupamento {} ".format(classe[0]))
    nomeFrame = 'frame{}'.format(j)
    pixels = np.zeros((1, 4), dtype=np.int8)
    img = cv2.imread("itai-sp.png")
    cv2.namedWindow(nomeFrame)
    cv2.setMouseCallback(nomeFrame, on_mouse)
    # Mostra a imagem e espera a tecla ser pressionada
    cv2.imshow(nomeFrame, img)
    cv2.waitKey(0)
    classe[0] += 1
    count[0] = 0
cv2.destroyAllWindows()