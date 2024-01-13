import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image in grayscale
img1 = cv.imread('beans_low_contrast.tif', cv.IMREAD_GRAYSCALE)

# Print the size and intensity range of the original image
print('Size of image is {}'.format(img1.shape))
print('Minimum and Maximum intensity value in image is ({},{})'.format(np.min(img1), np.max(img1)))

# Display the original image
plt.figure(figsize=(5, 5))
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Perform contrast stretching
min_intensity = np.min(img1)
max_intensity = np.max(img1)
L = 256  # Number of intensity levels for an 8-bit image

# Mapping function for contrast stretching
mapping_function = lambda r: ((r - min_intensity) / (max_intensity - min_intensity)) * (L - 1)

# Apply the mapping function to the image
img_stretched = mapping_function(img1)

# Display the contrast-stretched image
plt.figure(figsize=(5, 5))
plt.imshow(img_stretched, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Contrast Stretched Image')
plt.show()

# Print size and intensity range of the stretched image
print('Size of stretched image is {}'.format(img_stretched.shape))
print('Minimum and Maximum intensity value in stretched image is ({},{})'.format(np.min(img_stretched), np.max(img_stretched)))

# Display the contrast-stretching plot
stretch_plot = cv.imread('/content/Histogram_stretching_plot.png')
plt.figure(figsize=(5, 5))
plt.imshow(stretch_plot)
plt.axis('off')
plt.title('Contrast Stretching Plot')
plt.show()
