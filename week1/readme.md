# Image Enhancement Techniques

## Negative Transformation
This simple transformation results in an inverted representation of the original image.

## Example usage for Negative Transformation
img = cv.imread('your_image_path.jpg', cv.IMREAD_GRAYSCALE)
neg_img = L - 1 - img

## Contrast Stretching
Contrast stretching is applied to enhance the contrast of an image by stretching the intensity values over the entire range.

## Example usage for Contrast Stretching
min_intensity = np.min(img1)
max_intensity = np.max(img1)
mapping_function = lambda r: ((r - min_intensity) / (max_intensity - min_intensity)) * (L - 1)
img_stretched = mapping_function(img1)

## Power-law (Gamma) Transformation
A power-law (gamma) transformation is applied to adjust the gamma value, which controls the brightness and contrast of an image.

## Example usage for Power-law (Gamma) Transformation
gamma = 2.0
img_gamma_transformed = np.power(img3, gamma)



