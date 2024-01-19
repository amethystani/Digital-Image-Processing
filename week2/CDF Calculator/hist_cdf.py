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

# Calculate cdf
hist_cdf = np.cumsum(hist_pmf)

# Print and verify the result
print("Histogram CDF:")
print(hist_cdf)

# Plot the histogram and cdf
fig, ax1 = plt.subplots()

ax1.set_xlabel('Pixel Value')
ax1.set_ylabel('Frequency', color='b')
ax1.bar(bins[:-1], hist, color='b', alpha=0.7, label='Histogram')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.set_ylabel('CDF', color='r')
ax2.plot(bins[:-1], hist_cdf, color='r', label='CDF', linewidth=2)
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
plt.show()
