import sys 
import cv2 
import numpy as np 
 
 # Extrai o contorno das imagens
def get_all_contours(img): 
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return contours
 
if __name__=='__main__': 
    # le imagem com varios objetos 
    img = cv2.imread('random_shapes.png') 
 
    img_orig = np.copy(img) 
    input_contours = get_all_contours(img) 
    solidity_values = [] 
 
    # calcula a solidity (quão perto o objeto está do seu retangulo convexo)
    for contour in input_contours: 
        area_contour = cv2.contourArea(contour) 
        convex_hull = cv2.convexHull(contour) 
        area_hull = cv2.contourArea(convex_hull) 
        solidity = float(area_contour)/area_hull
        print(solidity)
        # cria vetor de solidity calculados
        solidity_values.append(solidity) 
 
    # Clustering using KMeans 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    flags = cv2.KMEANS_RANDOM_CENTERS 
    solidity_values = np.array(solidity_values).reshape((len(solidity_values),1)).astype('float32')
    '''
    compactness : It is the sum of squared distance from each point to their corresponding centers.
    labels : This is the label array (same as 'code' in previous article) where each element marked '0', '1'.....
    centers : This is array of centers of clusters.
'''
    compactness, labels, centers = cv2.kmeans(solidity_values, 2, None, criteria, 10, flags) 
    print(centers)
    closest_class = np.argmin(centers)
    print(closest_class)
    output_contours = []
    # closest_class é o label de menor valor de solidity.
    # então, as figuras que foram agrupadas com menor valor de solidity  vão ser "censuradas" 
    for i in solidity_values[labels==closest_class]: 
        index = np.where(solidity_values==i)[0][0] 
        output_contours.append(input_contours[index]) 
 
    cv2.drawContours(img, output_contours, -1, (0,0,0), 3) 
    cv2.imshow('Output', img) 
 
    # Censoring 
    for contour in output_contours: 
        rect = cv2.minAreaRect(contour) 
        box = cv2.boxPoints(rect) 
        box = np.int0(box) 
        cv2.drawContours(img_orig,[box],0, (0,0,0), -1) 
 
    cv2.imshow('Censored', img_orig) 
    cv2.waitKey(0) 
