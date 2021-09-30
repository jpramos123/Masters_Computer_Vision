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


sk_image = cv.imread('standard_test_images/house.tif', 0)
image = img_as_ubyte(sk_image)

H = cv.calcHist([image], [0], None, [256], [0,256])
theta = np.zeros(256)
Hf = np.zeros(256)
Hb = np.zeros(256)

for T in range(1,255):
    Hf[T] = - np.sum( entp(H[:T-1] / np.sum(H[1:T-1])) )
    Hb[T] = - np.sum( entp(H[T:] / np.sum(H[T:])) )
    theta[T] = Hf[T] + Hb[T]

theta_max = np.argmax(theta)

ret, im_shannon = cv.threshold(image, theta_max, 255, cv.THRESH_BINARY) 

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Imagem Original")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(im_shannon, cmap='gray')
ax1.set_title("Entropia")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()  