import numpy as np
import cv2
img = cv2.imread('amazonia2.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
final=np.concatenate((img,res2),axis=1)
cv2.imshow('res2',final)
num=center.shape[0]
k=np.ones((10,100,100,3),np.uint8)
for j in range(0,num):
  print("Centro do cluster %d:"%j)
  print(center[j])


k1= np.ones((100,100,3),np.uint8)
k2=np.ones((100,100,3),np.uint8)
k3=np.ones((100,100,3),np.uint8)
k1[j,:,:,0]=[center[0][0]]
k1[j,:,:,1]=[center[0][1]]
k1[j,:,:,2]=[center[0][2]]

k2[:,:,0]=[center[1][0]]
k2[:,:,1]=[center[1][1]]
k2[:,:,2]=[center[1][2]]

k3[:,:,0]=[center[2][0]]
k3[:,:,1]=[center[2][1]]
k3[:,:,2]=[center[2][2]]

final=np.concatenate((k1,k2,k3),axis=1)

cv2.imshow("cores dos clusters",final)
cv2.waitKey(0)
cv2.destroyAllWindows()









































