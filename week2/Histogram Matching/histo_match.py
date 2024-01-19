import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the reference and input images
ref0 = cv.imread('penstand_bright.jpg')
ref = cv.cvtColor(ref0, cv.COLOR_BGR2GRAY)

inp0 = cv.imread('penstand_lowlight2.jpg')
inp = cv.cvtColor(inp0, cv.COLOR_BGR2GRAY)

# Display the reference and input images
plt.figure(figsize=(20, 20))
plt.subplot(121)
plt.imshow(inp, cmap='gray', vmin=0, vmax=255)
plt.title('Input Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(ref, cmap='gray', vmin=0, vmax=255)
plt.title('Reference Image')
plt.axis('off')

plt.show()

# Histogram matching
def histogram_matching(input_image, reference_image):
    matched_image = np.zeros_like(input_image)

    # Calculate histograms
    hist_input, bins_input = np.histogram(input_image.flatten(), 256, [0, 256])
    hist_ref, bins_ref = np.histogram(reference_image.flatten(), 256, [0, 256])

    # Calculate cumulative distribution functions (CDF)
    cdf_input = np.cumsum(hist_input) / np.sum(hist_input)
    cdf_ref = np.cumsum(hist_ref) / np.sum(hist_ref)

    # Match histograms
    for i in range(256):
        idx = np.argmin(np.abs(cdf_input[i] - cdf_ref))
        matched_image[input_image == i] = idx

    return matched_image

# Perform histogram matching
matched_image = histogram_matching(inp, ref)

# Display the original input, reference, and matched images
plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(inp, cmap='gray', vmin=0, vmax=255)
plt.title('Input Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(ref, cmap='gray', vmin=0, vmax=255)
plt.title('Reference Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(matched_image, cmap='gray', vmin=0, vmax=255)
plt.title('Matched Image')
plt.axis('off')

plt.show()
