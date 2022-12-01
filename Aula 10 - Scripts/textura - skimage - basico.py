import skimage
import cv2

from skimage.io import imread
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
import warnings
warnings.filterwarnings("always")

import numpy as np
import pandas as pd

'''
image : array_like of uint8
Integer typed input image. The image will be cast to uint8, so the maximum value must be less than 256.

distances : array_like
List of pixel pair distance offsets.

angles : array_like
List of pixel pair angles in radians.

levels : int, optional
The input image should contain integers in [0, levels-1], where levels indicate the number of grey-levels counted (typically 256 for an 8-bit image). The maximum value is 256.

symmetric : bool, optional
If True, the output matrix P[:, :, d, theta] is symmetric. This is accomplished by ignoring the order of value pairs, so both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.

normed : bool, optional
If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset. The elements of the resulting matrix sum to 1. The default is False.
'''
caracteristicas=[]
#skimage.feature.graycomatrix(image, distances, angles, levels=256, symmetric=False, normed=False)

def extrairCaracteristicas(im):
    for j in  range (1,50,5):
        g = skimage.feature.graycomatrix(im, [j], [0], levels=256, symmetric=False, normed=True)
        contrast= skimage.feature.graycoprops(g, 'contrast')[0][0]
        energy= skimage.feature.graycoprops(g, 'energy')[0][0]
        homogeneity= skimage.feature.graycoprops(g, 'homogeneity')[0][0]
        correlation=skimage.feature.graycoprops(g, 'correlation')[0][0]
        dissimilarity=skimage.feature.graycoprops(g, 'dissimilarity')[0][0]
        ASM=skimage.feature.graycoprops(g, 'ASM')[0][0]


        #print('contrast is: ', contrast)
        #print('energy is: ', energy)
        #print('homogeneity is: ', homogeneity)
        #print('correlation is: ', correlation)
        #print('dissimilarity is: ', dissimilarity)
        #print('ASM is: ', ASM)
        #print("----------------------------\n")
        caracteristicas.append([contrast,energy,homogeneity,correlation,dissimilarity,ASM])

im = cv2.imread("tijolo1.png")
im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = skimage.img_as_ubyte(im)

extrairCaracteristicas(im)
#agora, com grama
im = cv2.imread("grama1.jpg")
im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = skimage.img_as_ubyte(im)
extrairCaracteristicas(im)

# pedras
im = cv2.imread("pedras1.jpg")
im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = skimage.img_as_ubyte(im)
extrairCaracteristicas(im)

df = pd.DataFrame(caracteristicas, columns = ['contrast','energy','homogeneity','correlation,','dissimilarity','ASM'])
df.head(30)
print(df)
