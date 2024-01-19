import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img0 = cv.imread('people_lowlight.jpg')
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

img_flat = img.ravel()
hist, bins = np.histogram(img_flat, 256, [0, 256])

# Calculate pmf
hist_pmf = hist / np.sum(hist)

# Calculate cdf
hist_cdf = np.cumsum(hist_pmf)

# Calculate T(r) using CDF as the transformation function
T_r = np.round(255 * hist_cdf).astype(int)

# Apply the transformation to the image
img_transformed = T_r[img]

# Plot the original and transformed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_transformed, cmap='gray', vmin=0, vmax=255)
plt.title('Transformed Image (T(r))')
plt.axis('off')

plt.show()
