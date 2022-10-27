import cv2
import numpy as np


image=cv2.imread('maoaberta.png')
cv2.imshow("original",image)
img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_img=cv2.Canny(img_gray,30,200)
contours, hierarchy=cv2.findContours(canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# lembrando que -1 plota todos os contornos
cv2.drawContours(image,contours,-1,(0,255,0),2)
image2=cv2.imread('maoaberta.png')
for c in contours:
    convexHull = cv2.convexHull(c)
    cv2.drawContours(image2, [convexHull], -1, (255, 0, 0), 2)
    hull = cv2.convexHull(c, returnPoints = False)
    defects = cv2.convexityDefects(c, hull)


for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    #print(s,e,f,d)
    #print(far)
    cv2.line(image2,start,end,[0,255,0],2)
    cv2.circle(image2,far,5,[0,0,255],-1)
cv2.imshow('ConvexHull', image2)
cv2.waitKey(0)
cv2.imwrite("teste.png",image2)
cv2.destroyAllWindows()
