from FilterImage import FilterImage

filters = FilterImage("aviao.jpg", True)
sourceImg = filters.getSourceImg()

# Aplica filtro Sobel e bitwise AND com a original
sobelImg = filters.applySobelFilter(1, 1, 5)
sobelMeanImg5x5 = filters.applyMeanFilter(sobelImg, (5, 5))
sobelAndSrcImg = filters.applyBitwiseAnd(sourceImg, sobelImg)
sobelMean5x5AndSrcImg = filters.applyBitwiseAnd(sourceImg, sobelMeanImg5x5)

# Aplica filtro Roberts e bitwise AND com a original
robertsImg = filters.applyRobertsFilter()
robertsMeanImg5x5 = filters.applyMeanFilter(robertsImg, (5, 5))
robertsAndSrcImg = filters.applyBitwiseAnd(sourceImg, robertsImg)
robertsMean5x5AndSrcImg = filters.applyBitwiseAnd(sourceImg, robertsMeanImg5x5)

# Aplica filtro Prewitt e bitwise AND com a original
prewittImg = filters.applyPrewittFilter()
prewittMeanImg5x5 = filters.applyMeanFilter(prewittImg, (5, 5))
prewittAndSrcImg = filters.applyBitwiseAnd(sourceImg, prewittImg)
prewittMean5x5AndSrcImg = filters.applyBitwiseAnd(sourceImg, prewittMeanImg5x5)

# Aplica filtro Canny e bitwise AND com a original
cannyImg = filters.applyCannyFilter(50, 100)
cannyMeanImg5x5 = filters.applyMeanFilter(cannyImg, (5, 5))
cannyAndSrcImg = filters.applyBitwiseAnd(sourceImg, cannyImg)
cannyMean5x5AndSrcImg = filters.applyBitwiseAnd(sourceImg, cannyMeanImg5x5)

outputDict = {
    "sobel"   : {
        "resultImg"             : sobelImg,
        "meanImg5x5"            : sobelMeanImg5x5,
        "sobelAndSrcImg"        : sobelAndSrcImg,
        "sobelMean5x5AndSrcImg" : sobelMean5x5AndSrcImg
    },
    "roberts" : {
        "resultImg"               : robertsImg,
        "meanImg5x5"              : robertsMeanImg5x5,
        "robertsAndSrcImg"        : robertsAndSrcImg,
        "robertsMean5x5AndSrcImg" : robertsMean5x5AndSrcImg
    },
    "prewitt" : {
        "resultImg"               : prewittImg,
        "meanImg5x5"              : prewittMeanImg5x5,
        "prewittAndSrcImg"        : prewittAndSrcImg,
        "prewittMean5x5AndSrcImg" : prewittMean5x5AndSrcImg
    },
    "canny"   : {
        "resultImg"             : cannyImg,
        "meanImg5x5"            : cannyMeanImg5x5,
        "cannyAndSrcImg"        : cannyAndSrcImg,
        "cannyMean5x5AndSrcImg" : cannyMean5x5AndSrcImg
    }
}
filters.generateOutput(outputDict)