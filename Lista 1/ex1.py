import cv2 as cv
from matplotlib import pyplot as plt

# Read image
img = cv.imread('standard_test_images/lena_color_256.tif',)

# Apply average filter once to the image
blur_img = cv.blur(img, (3,3))

# Loop and apply 10x the average filter in the image
blur_img_loop = img
blur_imgs_arr = []
for i in range(10):
    blur_img_loop = cv.blur(blur_img_loop, (3,3))

blur_imgs_arr.append(blur_img_loop)

# Loop and apply 20x the average filter in the image
blur_img_loop = img
for i in range(100):
    blur_img_loop = cv.blur(blur_img_loop, (3,3))

blur_imgs_arr.append(blur_img_loop)

# Plot everything to comparision
plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur_img),plt.title('Blurred Once')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(blur_imgs_arr[0]),plt.title('Blurred in Loop 10x')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(blur_imgs_arr[1]),plt.title('Blurred in Loop 100x')
plt.xticks([]), plt.yticks([])

plt.show()