import cv2
import numpy as np

pessoas = cv2.imread("4pessoas.jpg")
flor = cv2.imread("flor.jpg")
praia = cv2.imread("praia.jpg")

colagem = np.hstack([pessoas, flor, praia])
cv2.imwrite("cena.png", colagem)
cv2.imshow("colagem", colagem)
cv2.waitKey(0)
