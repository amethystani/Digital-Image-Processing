import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def gamma_transform(image, gamma):
    # Ensure input image is in floating-point format for calculations
    image = image.astype(float)

    # Apply gamma transformation formula
    transformed_image = np.power(image, gamma)

    # Scale the result to the valid pixel intensity range
    transformed_image = np.clip(transformed_image, 0, 255)

    # Convert back to uint8 format for display
    transformed_image = transformed_image.astype(np.uint8)

    return transformed_image


# Read the image in grayscale
img3 = cv.imread('washed_out_aerial_image.tif', cv.IMREAD_GRAYSCALE)

# Print the size and intensity range of the original image
print('Size of image is {}'.format(img3.shape))
print('Minimum and Maximum intensity value in image is ({},{})'.format(np.min(img3), np.max(img3)))

# Display the original image
plt.figure(figsize=(10, 10))
plt.imshow(img3, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Define gamma value
gamma = 0.8

# Perform gamma transformation using the function
img_gamma_transformed = gamma_transform(img3, gamma)

# Display the gamma-transformed image
plt.figure(figsize=(10, 10))
plt.imshow(img_gamma_transformed, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Gamma Transformed Image (γ = {})'.format(gamma))
plt.show()
