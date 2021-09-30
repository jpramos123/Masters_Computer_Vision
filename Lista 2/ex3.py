import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('standard_test_images/house.tif')
pixel_vals = image.reshape((-1,3))
print(pixel_vals)
pixel_vals = np.float32(pixel_vals)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
ks = [3,6]
seg_images = []
for k in ks:
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    

    segmented_image = segmented_data.reshape((image.shape))
    seg_images.append(segmented_image)


titles = ['Imagem Original', 'Segmentação com K-Means (k=3)', 'Segmentação com K-Means (k=6)']
images = [image, seg_images[0], seg_images[1]]

for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()