import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.io import imread


def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp


img = cv.imread('standard_test_images/house.tif')


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

pixel_vals = img.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 3

retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

alt_image = segmented_data.reshape((img.shape))

#remove dimension
alt_image = alt_image[:, :, 0]
alt_image = img_as_ubyte(alt_image)


alt_image = cv.adaptiveThreshold(alt_image,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)

H = cv.calcHist([alt_image], [0], None, [256], [0,256])
theta = np.zeros(256)
Hf = np.zeros(256)
Hb = np.zeros(256)

for T in range(1,255):
    Hf[T] = - np.sum( entp(H[:T-1] / np.sum(H[1:T-1])) )
    Hb[T] = - np.sum( entp(H[T:] / np.sum(H[T:])) )
    theta[T] = Hf[T] + Hb[T]

theta_max = np.argmax(theta)

ret, alt_image = cv.threshold(alt_image, theta_max, 255, cv.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
alt_image = cv.erode(alt_image, kernel, iterations=5)
alt_image = cv.dilate(alt_image, kernel, iterations=5)

img0 = ax0.imshow(img, cmap=plt.cm.gray)
ax0.set_title("Imagem Original")
ax0.axis("off")

img1 = ax1.imshow(alt_image, cmap='gray')
ax1.set_title("Imagem com erosão e dilatação aplicadas (2 Iterações)")
ax1.axis("off")

fig.tight_layout()

plt.show()  