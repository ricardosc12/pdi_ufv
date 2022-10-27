import sys 
import cv2 
import numpy as np 
 
# Extract all the contours from the image 
def get_all_contours(img): 
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0) 
    # Find all the contours in the thresholded image. The values 
    # for the second and third parameters are restricted to a 
    # certain number of possible values. You can learn more 'findContours' function here: 
    # http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return contours

# Extract reference contour from the image 
def get_ref_contour(img): 
    contours = get_all_contours(img)

    # Extract the relevant contour based on area ratio. We use the 
    # area ratio because the main image boundary contour is 
    # extracted as well and we don't want that. This area ratio 
    # threshold will ensure that we only take the contour inside the image. 
    for contour in contours: 
        area = cv2.contourArea(contour) 
        img_area = img.shape[0] * img.shape[1] 
        if 0.05 < area/float(img_area) < 0.8: 
            return contour 
 
if __name__=='__main__': 
    # imagem a ser encontrada
    img1 = cv2.imread('boomerang.png')
    cv2.imshow('Imagem a ser encontrada',img1)
    cv2.waitKey(0)
 
    # imagem com varias formas. VAMOS TENTAR ENCONTRAR A IMAGEM ACIMA NA IMAGEM ABAIXO
    img2 = cv2.imread('random_shapes.png') 
 
    # RETORNANDO CONTORNOS DA IMAGEM REFERENCIA
    ref_contour = get_ref_contour(img1)
 
    # EXTRAINDO CONTORNO DE TODOS OS OBJETOS NA IMAGEM 2
    input_contours = get_all_contours(img2) 
 
    closest_contour = None
    min_dist = None
    contour_img = img2.copy()

    # ENCONTRANDO O  CONTORNO MAIS PRÓXIMO 
    for i, contour in enumerate(input_contours): 
        # TENTA ENCONTRAR UM CONTORNO MAIS PROXIMO DA IMAGEM PROCURADA
        # uTILIZA OS MOMENTOS DE HU PARA ISTO, ENTÃO SÃO INVARIANTES A ESCALA, ROTAÇÃO E TRANSLAÇÃO
        # Comparison method CV_CONTOURS_MATCH_I3 (second argument)(https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaf2b97a230b51856d09a2d934b78c015f)
        ret = cv2.matchShapes(ref_contour, contour, 3, 0.0)
        print("CONTORNO  %d SE PARECE COM A IMAGEM PROCURARA EM  %f" % (i, ret))
        if min_dist is None or ret < min_dist:  # o menor valor representa o contorno mais proximo ao contorno da imagem original
            min_dist = ret 
            closest_contour = contour
            menor=i
 
    cv2.drawContours(img2, [closest_contour], 0 , color=(0,0,0), thickness=3) 
    cv2.imshow('MELHOR CONTORNO ENCONTRADO ', img2)
    print("menor contorno: # %d" % menor)
    cv2.waitKey() 
