import cv2
import numpy as np
from matplotlib import pyplot as plt

# Histograma
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem em escala de cinza", img)
cv2.waitKey(0)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histograma')
plt.show()
cv2.waitKey(0)

# Transformação linear (alpha varia o contraste e beta varia o brilho)
alpha = 1.2 # 1 a 3
beta = 0 # 0 a 100
newImage = np.zeros(img.shape, img.dtype)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        newImage[y,x] = np.clip(alpha * img[y, x] + beta, 0, 255)
cv2.imshow("Nova imagem lin", newImage)
cv2.waitKey(0)
plt.hist(newImage.ravel(), 256, [0, 256])
plt.title('Hist. com transf. linear')
plt.show()
cv2.waitKey(0)

# Transformação logarítmica (alpha varia o contraste e beta varia o brilho)
alpha = 25.70
beta = 1
newImage = np.zeros(img.shape, img.dtype)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        newImage[y,x] = np.clip(alpha * np.log2(img[y, x] + beta), 0, 255)
cv2.imshow("Nova imagem log", newImage)
cv2.waitKey(0)
plt.hist(newImage.ravel(), 256, [0, 256])
plt.title('Hist. com transf. logaritmica')
plt.show()
cv2.waitKey(0)

# Equalização
hist, _ = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
normalizedCdf = cdf * hist.max() / cdf.max()
plt.subplot(221)
plt.plot(normalizedCdf, color = 'b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histograma'), loc='upper left')
plt.title('Hist. da img. original'), plt.xticks([]), plt.yticks([])

equalizeImg = cv2.equalizeHist(img)
hist, _ = np.histogram(equalizeImg.flatten(), 256, [0, 256])
cdf = hist.cumsum()
normalizedCdf = cdf * hist.max() / cdf.max()
plt.subplot(222)
plt.plot(normalizedCdf, color='b')
plt.hist(equalizeImg.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histograma'), loc='upper left')
plt.title('Hist. da img. equalizada'), plt.xticks([]), plt.yticks([])

plt.subplot(223)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv2.cvtColor(equalizeImg, cv2.COLOR_BGR2RGB))
plt.show()

# Limiarização por OTSU e métodos da openCV
threshold = 0
maxValue = 255

_, binaria = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binariaInv = cv2.threshold(img, threshold, maxValue, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, tozero = cv2.threshold(img, threshold, maxValue, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
_, tozeroInv = cv2.threshold(img, threshold, maxValue, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
_, truncImg = cv2.threshold(img, threshold, maxValue, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

images = [img, binaria, binariaInv, tozero, tozeroInv, truncImg]
titles = ["Original", "Binaria", "Binaria invertida", "Tozero", "TOZERO INV", "Trunc"]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
plt.show()
