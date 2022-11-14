import cv2

img = cv2.imread("56.png", 0)
bitwise_NOT = cv2.bitwise_not(img)
s,imagem = cv2.threshold(bitwise_NOT , 60, 255, cv2.THRESH_BINARY)
cv2.imshow('NOT',bitwise_NOT)
cv2.waitKey(0)
cv2.imshow('LOL', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()