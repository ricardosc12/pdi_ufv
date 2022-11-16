from numpy.linalg import norm
from scipy import misc
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
import numpy as np
import mahotas
import pickle

class ZernikeMoments:
    def __init__(self, radius):
        self.radius = radius # Size of radius used when computing moments
    
    def describe(self, image): # Returns ZMoments of image
        return mahotas.features.zernike_moments(image, self.radius)
class Searcher:
    # Initializes with an index of sprite Zernike moment features
    # Querying an image's features against it causes it to find the
    # sprite in the database that's nearest
    def __init__(self, index):
        self.index = index
        
    def search(self, queryFeatures):
        results = {}
        for (key, features) in self.index.items():
            distance = norm(queryFeatures - features)
            results[key] = distance
            
        results = sorted([(v, k) for (k, v) in results.items()])
        return results # Nearest sprite is first in list
index = open('pokesprite_index', 'rb')
index = pickle.load(index)
#img = misc.imread('./result_img/ditto.jpg')

img = cv2.imread('pokemon_rb_imgs/ditto.jpg')
cv2.imshow("Pokemon Procurado",img)
cv2.waitKey(0)

img = (resize(img, [56, 56])*255).astype('uint8')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# apply adaptive thresholding with OpenCV
neighbourhood_size = 25
constant_c = 15
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                               neighbourhood_size, constant_c)




#adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)

plt.imshow(thresh, cmap='gray');
cv2.imshow("aida",thresh)
cv2.waitKey(0)




#Adaptive Thresholding
###Adaptive thresholding uses a locally calculated threshold, typically the mean Gaussian-weighted sum of the neighboring pixels, shifted by some constant.

#In OpenCV, block_size determines the neighborhood area, and C is the constant by which to translate the thresholds calculated at each pixel.

#If we don't shift the threshold using C, then each pixel will be white if it's larger than the average of its neighbors, or black if it isn't.

#If the neighbors are similar, but not quite the same (as in the case above where the screen was uneven), then by setting C to the range of the neighbors, we can ensure that the threshold will classify the neighbors in the same way.

#A larger neighborhood will be less sensitive to rapid variations in intensity, whereas a smaller one might be insufficiently sensitive, causing small genuinely dark regions to be thresholded high.

outline = np.zeros(img.shape, dtype='float32')
cnts, _ = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
extCnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
outline = cv2.drawContours(outline, [extCnt], -1, 255, -1)
print("outline")
cv2.imshow("outline",outline)
cv2.waitKey(0)


outline=np.uint8(outline)
outline = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)


# apply adaptive thresholding with OpenCV
neighbourhood_size = 25
constant_c = 15
outline  = cv2.threshold(outline,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)





desc = ZernikeMoments(21) # Initialize ZM instance
#queryFeatures = desc.describe(outline) # Get 21 ZM features
queryFeatures=mahotas.features.zernike_moments(outline, 10)
searcher = Searcher(index) # Initialize db
results = searcher.search(queryFeatures) # Return list of nearest images
print('It\'s a %s!' % results[0][1])

results

