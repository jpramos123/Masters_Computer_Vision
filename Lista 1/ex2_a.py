import cv2 as cv
from matplotlib import pyplot as plt

scale = 1
delta = 0
ddepth = cv.CV_16S

img = cv.imread('chessboard.png')


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

plt.subplot(221),plt.hist(grad.ravel(), 256, [0,256]),plt.title('Sobel')
plt.xticks([]), plt.yticks([])
plt.show()

    
