import cv2 as cv
from matplotlib import pyplot as plt

ddepth = cv.CV_16S
kernel_size = 3

img = cv.imread('chessboard.png')


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

dst = cv.Laplacian(img, ddepth, ksize=kernel_size)

abs_dst = cv.convertScaleAbs(dst)


plt.subplot(221),plt.hist(abs_dst.ravel(), 256, [0,256]),plt.title('Laplace')
plt.xticks([]), plt.yticks([])
plt.show()

    
