import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img0 = cv.imread('people_lowlight.jpg')
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

img_flat = img.ravel()
hist, bins = np.histogram(img_flat, 256, [0, 256])

# Calculate pmf
hist_pmf = hist / np.sum(hist)

# Print and verify the result
print("Histogram PMF:")
print(hist_pmf)

# Plot the histogram
plt.bar(bins[:-1], hist, color='b', alpha=0.7, label='Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
