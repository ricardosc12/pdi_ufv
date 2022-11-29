import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread('./stack.png',0)


blur = cv2.GaussianBlur(img, (7,7), 0)

brightness = 50
contrast = 180
blur = np.int16(blur)
blur = blur * (contrast/127+1) - contrast + brightness
blur = np.clip(blur, 0, 255)
blur = np.uint8(blur)

# element = cv2.getStructuringElement(0, (2*20 + 1, 2*20+1), (20, 20))
# dilatation_dst = cv2.dilate(blur, element)

# element = cv2.getStructuringElement(2, (2*20 + 1, 2*20+1), (20, 20))
# erosion_dst = cv2.erode(dilatation_dst, element)

ret,thresh3 = cv2.threshold(blur,140,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

image = np.hstack([img,blur,thresh3])


cv2.imshow("teste",image)
cv2.waitKey(0)