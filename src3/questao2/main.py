from NoiseFilter import NoiseFilter

filters = NoiseFilter("aviao.jpg", True)
sourceImg = filters.getSourceImg()

# Aplica os ruidos salt&pepper e gaussiano
saltAndPepperImg = filters.applySaltAndPepperNoise(0.5, 0.04)
gaussianImg = filters.applyGaussianNoise(0)

# Aplica o filtro de media 3x3 nas tres imagens
sourceMeanImg3x3 = filters.applyMeanFilter(sourceImg, (3, 3))
saltAndPepperMeanImg3x3 = filters.applyMeanFilter(saltAndPepperImg, (3, 3))
gaussianMeanImg3x3 = filters.applyMeanFilter(gaussianImg, (3, 3))
# Aplica o filtro de media 7x7 nas tres imagens
sourceMeanImg7x7 = filters.applyMeanFilter(sourceImg, (7, 7))
saltAndPepperMeanImg7x7 = filters.applyMeanFilter(saltAndPepperImg, (7, 7))
gaussianMeanImg7x7 = filters.applyMeanFilter(gaussianImg, (7, 7))

# Aplica o filtro de mediana 3x3 nas tres imagens
sourceMedianImg3x3 = filters.applyMedianFilter(sourceImg, 3)
saltAndPepperMedianImg3x3 = filters.applyMedianFilter(saltAndPepperImg, 3)
gaussianMedianImg3x3 = filters.applyMedianFilter(gaussianImg, 3)
# Aplica o filtro de mediana 7x7 nas tres imagens
sourceMedianImg7x7 = filters.applyMedianFilter(sourceImg, 7)
saltAndPepperMedianImg7x7 = filters.applyMedianFilter(saltAndPepperImg, 7)
gaussianMedianImg7x7 = filters.applyMedianFilter(gaussianImg, 7)

# Aplica o filtro gaussiano 3x3 nas tres imagens
sourceGaussianImg3x3 = filters.applyGaussianFilter(sourceImg, (3, 3))
saltAndPepperGaussianImg3x3 = filters.applyGaussianFilter(saltAndPepperImg, (3, 3))
gaussianGaussianImg3x3 = filters.applyGaussianFilter(gaussianImg, (3, 3))
# Aplica o filtro gaussiano 5x5 nas tres imagens
sourceGaussianImg5x5 = filters.applyGaussianFilter(sourceImg, (5, 5))
saltAndPepperGaussianImg5x5 = filters.applyGaussianFilter(saltAndPepperImg, (5, 5))
gaussianGaussianImg5x5 = filters.applyGaussianFilter(gaussianImg, (5, 5))

# Gera os histogramas das imagens geradas
sourceHistogram = filters.generateHistogram(sourceImg)
saltAndPepperHistogram = filters.generateHistogram(saltAndPepperImg)
gaussianHistogram = filters.generateHistogram(gaussianImg)

sourceMean3x3Histogram = filters.generateHistogram(sourceMeanImg3x3)
saltAndPepperMean3x3Histogram = filters.generateHistogram(saltAndPepperMeanImg3x3)
gaussianMean3x3Histogram = filters.generateHistogram(gaussianMeanImg3x3)

sourceMean7x7Histogram = filters.generateHistogram(sourceMeanImg7x7)
saltAndPepperMean7x7Histogram = filters.generateHistogram(saltAndPepperMeanImg7x7)
gaussianMean7x7Histogram = filters.generateHistogram(gaussianMeanImg7x7)

sourceMedian3x3Histogram = filters.generateHistogram(sourceMedianImg3x3)
saltAndPepperMedian3x3Histogram = filters.generateHistogram(saltAndPepperMedianImg3x3)
gaussianMedian3x3Histogram = filters.generateHistogram(gaussianMedianImg3x3)

sourceMedian7x7Histogram = filters.generateHistogram(sourceMedianImg7x7)
saltAndPepperMedian7x7Histogram = filters.generateHistogram(saltAndPepperMedianImg7x7)
gaussianMedian7x7Histogram = filters.generateHistogram(gaussianMedianImg7x7)

sourceGaussian3x3Histogram = filters.generateHistogram(sourceGaussianImg3x3)
saltAndPepperGaussian3x3Histogram = filters.generateHistogram(saltAndPepperGaussianImg3x3)
gaussianGaussian3x3Histogram = filters.generateHistogram(gaussianGaussianImg3x3)

sourceGaussian5x5Histogram = filters.generateHistogram(sourceGaussianImg5x5)
saltAndPepperGaussian5x5Histogram = filters.generateHistogram(saltAndPepperGaussianImg5x5)
gaussianGaussian5x5Histogram = filters.generateHistogram(gaussianGaussianImg5x5)

outputDict = {
    "base"         : {
        "srcImg"   : sourceImg,
        "sEpImg"   : saltAndPepperImg,
        "gauImg"   : gaussianImg,
        "srcHist"  : sourceHistogram,
        "sEpHist"  : saltAndPepperHistogram,
        "gauHist"  : gaussianHistogram
    },
    "media3x3"     : {
        "srcImg"   : sourceMeanImg3x3,
        "sEpImg"   : saltAndPepperMeanImg3x3,
        "gauImg"   : gaussianMeanImg3x3,
        "srcHist"  : sourceMean3x3Histogram,
        "sEpHist"  : saltAndPepperMean3x3Histogram,
        "gauHist"  : gaussianMean3x3Histogram
    },
    "media7x7"     : {
        "srcImg"   : sourceMeanImg7x7,
        "sEpImg"   : saltAndPepperMeanImg7x7,
        "gauImg"   : gaussianMeanImg7x7,
        "srcHist"  : sourceMean7x7Histogram,
        "sEpHist"  : saltAndPepperMean7x7Histogram,
        "gauHist"  : gaussianMean7x7Histogram
    },
    "mediana3x3"   : {
        "srcImg"   : sourceMedianImg3x3,
        "sEpImg"   : saltAndPepperMedianImg3x3,
        "gauImg"   : gaussianMedianImg3x3,
        "srcHist"  : sourceMedian3x3Histogram,
        "sEpHist"  : saltAndPepperMedian3x3Histogram,
        "gauHist"  : gaussianMedian3x3Histogram
    },
    "mediana7x7"   : {
        "srcImg"   : sourceMedianImg7x7,
        "sEpImg"   : saltAndPepperMedianImg7x7,
        "gauImg"   : gaussianMedianImg7x7,
        "srcHist"  : sourceMedian7x7Histogram,
        "sEpHist"  : saltAndPepperMedian7x7Histogram,
        "gauHist"  : gaussianMedian7x7Histogram
    },
    "gaussiano3x3" : {
        "srcImg"   : sourceGaussianImg3x3,
        "sEpImg"   : saltAndPepperGaussianImg3x3,
        "gauImg"   : gaussianGaussianImg3x3,
        "srcHist"  : sourceGaussian3x3Histogram,
        "sEpHist"  : saltAndPepperGaussian3x3Histogram,
        "gauHist"  : gaussianGaussian3x3Histogram
    },
    "gaussiano5x5" : {
        "srcImg"   : sourceGaussianImg5x5,
        "sEpImg"   : saltAndPepperGaussianImg5x5,
        "gauImg"   : gaussianGaussianImg5x5,
        "srcHist"  : sourceGaussian5x5Histogram,
        "sEpHist"  : saltAndPepperGaussian5x5Histogram,
        "gauHist"  : gaussianGaussian5x5Histogram
    }
}
filters.generateOutput(outputDict)