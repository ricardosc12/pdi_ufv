import cv2 
import numpy as np
img1 = cv2.imread('s0.png',0)
img2 = cv2.imread('s1.png',0)

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)

cnts1, hierarchy  = cv2.findContours(thresh,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
cnts2, hierarchy  = cv2.findContours(thresh2,
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

cnt1 = cnts1[0]
cnt2 = cnts2[0]

cv2.drawContours(img1, cnt1, 5, (0, 255, 0), 2)
cv2.imshow("Imagem",img1)
cv2.waitKey(0)
ret = cv2.matchShapes(cnt1,cnt1,1,0.0)
print("(1,1): {:3.5f}".format(ret) )
ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print( "(1,2): {:3.5f}".format(ret) )
