import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.io import imread

img1 = cv.imread('standard_test_images/house.tif')
img2 = cv.imread('standard_test_images/jetplane.tif')
img1_gray = gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


H1 = cv.calcHist([img1_gray], [0], None, [256], [0,256])
H2 = cv.calcHist([img2_gray], [0], None, [256], [0,256])

print(cv.compareHist(H1, H2, cv.HISTCMP_KL_DIV))
plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(H1)
plt.show()