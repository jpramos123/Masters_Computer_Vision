import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('standard_test_images/house.tif', 0)

th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)

titles = ['Imagem Original', 'Limiar adaptativo']

images = [img, th]

for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()