from numpy.linalg import norm
from scipy.image import imread
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
import numpy as np
import mahotas

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
index = open('./pokesprite_index', 'rb')
index = pickle.load(index)

img = imread('./result_img/ditto.jpg')
img = (resize(img, [56, 56])*255).astype('uint8')

thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 43, 21)
plt.imshow(thresh, cmap='gray');

#Adaptive Thresholding
###Adaptive thresholding uses a locally calculated threshold, typically the mean Gaussian-weighted sum of the neighboring pixels, shifted by some constant.

#In OpenCV, block_size determines the neighborhood area, and C is the constant by which to translate the thresholds calculated at each pixel.

#If we don't shift the threshold using C, then each pixel will be white if it's larger than the average of its neighbors, or black if it isn't.

#If the neighbors are similar, but not quite the same (as in the case above where the screen was uneven), then by setting C to the range of the neighbors, we can ensure that the threshold will classify the neighbors in the same way.

#A larger neighborhood will be less sensitive to rapid variations in intensity, whereas a smaller one might be insufficiently sensitive, causing small genuinely dark regions to be thresholded high.

outline = np.zeros(img.shape, dtype='float32')
cnts, _ = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL,
                          method=cv2.CHAIN_APPROX_SIMPLE)
extCnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
outline = cv2.drawContours(outline, [extCnt], -1, 255, -1)
plt.imshow(outline, cmap='gray')

desc = ZernikeMoments(21) # Initialize ZM instance
queryFeatures = desc.describe(outline) # Get 21 ZM features

searcher = Searcher(index) # Initialize db
results = searcher.search(queryFeatures) # Return list of nearest images
print('It\'s a %s!' % results[0][1])

results
