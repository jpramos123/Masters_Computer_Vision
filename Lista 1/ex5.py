import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def calcFourier(img):
    print("Calculando Fourier")
    img = cv.imread(img, 0)
    dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return magnitude_spectrum

def clrHighFreq(img):
    print("Podando alta frequencia")
    img = cv.imread(img, 0)
    blur_img = cv.blur(img, (5,5))
    return blur_img

def clrLowFreq(img):
    print("Podando baixa frequencia")
    img = cv.imread(img)
    ddepth = cv.CV_16S
    kernel_size = 3
    la_place_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(la_place_img, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst

def calcAvg(img):
    print("Aplicando Filtro da Média")
    img = cv.imread(img, 0)
    blur_img = cv.blur(img, (5,5))
    return blur_img

def calcMedian(img):
    print("Aplicando Filtro da Mediana")
    img = cv.imread(img, 0)
    blur_img = cv.medianBlur(img, 5)
    return blur_img
 
def calcSobel(img):
    print("Aplicando FIltro Sobel")
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def calcSobelX(img):
    print("Calculando Gradiente em X")
    scale = 1
    delta = 0
    ddepth = cv.CV_64F

    img = cv.imread(img)

    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    return grad_x

def calcSobelY(img):
    print("Calculando Gradiente em Y")
    scale = 1
    delta = 0
    ddepth = cv.CV_64F

    img = cv.imread(img)

    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    return grad_y  

def calcSobelXAbs(img):
    print("Calculando Gradiente Absoluto em X")
    scale = 1
    delta = 0
    ddepth = cv.CV_64F

    img = cv.imread(img)

    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)

    return abs_grad_x

def calcSobelYAbs(img):
    print("Calculando Gradiente Absoluto em Y")
    scale = 1
    delta = 0
    ddepth = cv.CV_64F

    img = cv.imread(img)

    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_y = cv.convertScaleAbs(grad_y)
    return abs_grad_y

def calcGaussianFilter(img):
    print("Calculando Filtro Gaussiano")

    img = cv.imread(img)
    blur = cv.GaussianBlur(img,(5,5),0)

    return blur

src_img = cv.imread('standard_test_images/lake.tif')
fourier = calcFourier('standard_test_images/lake.tif')
highFreq = clrHighFreq('standard_test_images/lake.tif')
lowFreq = clrLowFreq('standard_test_images/lake.tif')
avg = calcAvg('standard_test_images/lake.tif')
median = calcMedian('standard_test_images/lake.tif')
sobel = calcSobel('standard_test_images/lake.tif')
sobelX = calcSobelX('standard_test_images/lake.tif')
sobelY = calcSobelY('standard_test_images/lake.tif')
sobelAbsX = calcSobelXAbs('standard_test_images/lake.tif')
sobelAbsY = calcSobelYAbs('standard_test_images/lake.tif')
gauss = calcGaussianFilter('standard_test_images/lake.tif')



plt.subplot(321),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(322),plt.imshow(fourier, cmap = 'gray'),plt.title('Espectro de Fourier'),
plt.xticks([]), plt.yticks([])

plt.subplot(323),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(highFreq, cmap = 'gray'),plt.title('Poda de Alta Frequencia'), 
plt.xticks([]), plt.yticks([])

plt.subplot(325),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(326),plt.imshow(lowFreq, cmap = 'gray'), plt.title('Poda de Baixa Frequencia'), 
plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(321),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(322),plt.imshow(avg, cmap = 'gray'), plt.title('Filtro da Média'), 
plt.xticks([]), plt.yticks([])

plt.subplot(323),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(median, cmap = 'gray'), plt.title('Filtro da Mediana'), 
plt.xticks([]), plt.yticks([])

plt.subplot(325),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(326),plt.imshow(sobel, cmap = 'gray'), plt.title('Filtro Sobel'), 
plt.xticks([]), plt.yticks([])

plt.show()  

plt.subplot(321),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(322),plt.imshow(sobelX, cmap = 'gray'), plt.title('Gradiente X'), 
plt.xticks([]), plt.yticks([])

plt.subplot(323),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(sobelY, cmap = 'gray'), plt.title('Gradiente Y'), 
plt.xticks([]), plt.yticks([])

plt.subplot(325),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(326),plt.imshow(sobelAbsX, cmap = 'gray'), plt.title('Gradiente Abs X'), 
plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(321),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(322),plt.imshow(sobelAbsY, cmap = 'gray'), plt.title('Gradiente Abs Y'), 
plt.xticks([]), plt.yticks([])

plt.subplot(323),plt.imshow(src_img, cmap = 'gray'),plt.title('Input Image'), 
plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(gauss, cmap = 'gray'), plt.title('Filtro Gaussiano'), 
plt.xticks([]), plt.yticks([])

plt.show()