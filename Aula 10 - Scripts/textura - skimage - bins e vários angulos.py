import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import skimage

import cv2 
from matplotlib import pyplot as plt 
img = io.imread('pedras1.jpg')

gray = color.rgb2gray(img)
cv2.imshow("Original",gray)
cv2.waitKey(0)
image = img_as_ubyte(gray)


histr = cv2.calcHist([image],[0],None,[256],[0,256]) 
plt.plot(histr) 
plt.show() 

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
inds = np.digitize(image, bins)

indsHistograma=bins.shape[0]*np.uint8(inds)
cv2.imshow("Com binds",indsHistograma)
cv2.waitKey(0)
histr = cv2.calcHist([indsHistograma],[0],None,[256],[0,256]) 
plt.plot(histr) 
plt.show() 

max_value = inds.max()+1
# alteramos o levels para o maximo do bins e usamos as 4 direções
matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

# GLCM properties
def contrast_feature(matrix_coocurrence):
	contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
	return "Contrast = ", contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')	
	return "Dissimilarity = ", dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = skimage.feature.graycoprops(matrix_coocurrence, 'homogeneity')
	return "Homogeneity = ", homogeneity

def energy_feature(matrix_coocurrence):
	energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
	return "Energy = ", energy

def correlation_feature(matrix_coocurrence):
	correlation = skimage.feature.graycoprops(matrix_coocurrence, 'correlation')
	return "Correlation = ", correlation

def asm_feature(matrix_coocurrence):
	asm = skimage.feature.graycoprops(matrix_coocurrence, 'ASM')
	return "ASM = ", asm

print(contrast_feature(matrix_coocurrence))
print(dissimilarity_feature(matrix_coocurrence))
print(homogeneity_feature(matrix_coocurrence))
print(energy_feature(matrix_coocurrence))
print(correlation_feature(matrix_coocurrence))
print(asm_feature(matrix_coocurrence))
