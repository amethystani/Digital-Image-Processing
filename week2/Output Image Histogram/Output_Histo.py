import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def linear_transformation(r, a=1, b=0):
    return np.clip(a * r + b, 0, 255).astype(int)

img0 = cv.imread('people_lowlight.jpg')
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

plt.figure(figsize=(15, 5))

# Calculate histogram for the original image
img_flat = img.ravel()
hist, bins = np.histogram(img_flat, 256, [0, 256])

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.bar(bins[:-1], hist, color='b', alpha=0.7, label='Original Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Original Image Histogram')

# Apply linear transformation T(r)
T_r = np.round(255 * np.cumsum(hist) / np.sum(hist)).astype(int)
out_img1 = T_r[img]
out_img_flat = out_img1.ravel()

# Calculate histogram for the transformed image
hist1, bins1 = np.histogram(out_img_flat, 256, [0, 256])

plt.subplot(1, 3, 3)
plt.imshow(out_img1, cmap='gray', vmin=0, vmax=255)
plt.title('Transformed Image (T(r))')
plt.axis('off')

plt.figure(figsize=(15, 5))
plt.bar(bins1[:-1], hist1, color='r', alpha=0.7, label='Transformed Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Transformed Image Histogram')

plt.show()
