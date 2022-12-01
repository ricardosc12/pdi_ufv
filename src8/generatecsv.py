import cv2
import numpy as np

classe = 0
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Captura um quadrado de pixel
        global pixels
        for i in range(-10, 9, 1):
            for j in range(-10, 9, 1):
                (B, G, R) = img[y+j, x+i]
                print([B, G, R, classe])
                p=np.array([[B, G, R, classe]])
                pixels=np.concatenate((pixels, p))
        
arquivo = "dados.csv"
numClasses = 5

for j in range(0, numClasses):
    print("Coletando para o agrupamento {} ".format(j))
    nomeFrame = 'frame{}'.format(j)
    pixels = np.zeros((1, 4), dtype=np.int8)
    img = cv2.imread("itai-sp.png")
    cv2.namedWindow(nomeFrame)
    cv2.setMouseCallback(nomeFrame, on_mouse)
    while True:
        # Mostra a imagem e espera a tecla ser pressionada
        cv2.imshow(nomeFrame, img)
        key = cv2.waitKey(1) & 0xFF
        # Se a tecla 'c' é pressionada sai do loop
        if key == ord("c"):
            with open(arquivo, 'a') as f:
                f.write("\n")
            break
    classe += 1
    pixels = pixels[1:] # Remove 000 inicial
    csv_rows = (["{},{},{},{}".format(i, j, k, l) for i, j, k, l in pixels])
    csv_text = "\n".join(csv_rows)
    with open(arquivo, 'a') as f:
        f.write(csv_text)
f.close()
cv2.destroyAllWindows()