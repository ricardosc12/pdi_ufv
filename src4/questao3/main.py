# Extract an object in HSV color space based on hue
import cv2 as cv2
import numpy as np

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def trackbar(x):

    hue = cv2.getTrackbarPos('hue:', window_name)
    saturation = cv2.getTrackbarPos('saturation:', window_name)
    value = cv2.getTrackbarPos('value:', window_name)


    lower = (hue-3, saturation-3, value-3)
    upper = (hue+3, saturation+3, value+3)

    mask = cv2.inRange(hsv, lower, upper)
    img2 = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(window_name, np.vstack([img, img2]))



img = cv2.imread('assets/cubos.png')

# Criando as duas imagens para comparação
img_clara = apply_brightness_contrast(img,120,50)
img_escura = apply_brightness_contrast(img,-140,-10)

# Concatenando
img = np.vstack([img, img_clara, img_escura])

# Lendo nos canais de HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
window_name = 'HSV'

cv2.imshow(window_name, img)

# Adicionando controle de hue, saturation e value para análise de diferenças

cv2.createTrackbar('hue:', window_name, 0, 180, trackbar)
cv2.createTrackbar('saturation:', window_name, 0, 255, trackbar)
cv2.createTrackbar('value:', window_name, 0, 255, trackbar)

# Com a mudança do brilho, os valores de value compreenderão nos espaços mais claros ou escuros, cujo as imagens produziram.
# Em saturação e value máximo, podemos transitar entre as cores mudando apenas o hue, que dita a tonalidade.

cv2.waitKey(0)
cv2.destroyAllWindows()
