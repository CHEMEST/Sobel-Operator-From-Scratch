import cv2
import numpy as np

src = 'you/filepath/here!'
fresh_img = cv2.imread(src, cv2.IMREAD_COLOR)
input_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)


def convolve2d(array, kernel):
    # Define
    array_height, array_width = array.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the dimensions of the output array
    output_height = array_height - kernel_height + 1
    output_width = array_width - kernel_width + 1
    
    # Initialize the output array with zeros
    output = np.zeros((output_height, output_width))
    
    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(array[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

# Blur
input_img = cv2.GaussianBlur(input_img, (7, 7), 0)

# Sobel kernels
x_axis_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

y_axis_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32)

# Perform convolution
x_edges = convolve2d(input_img, x_axis_kernel)
y_edges = convolve2d(input_img, y_axis_kernel)

# Calculate gradient magnitude
xE = x_edges ** 2
yE = y_edges ** 2
result = np.sqrt(xE + yE)

# Normalize
result = (result / np.max(result) * 255).astype(np.uint8)

# Calculate gradient direction
direction = np.arctan2(y_edges, x_edges)
direction = (direction + np.pi) / (2 * np.pi) * 255  # Normalize to range [0, 255]
direction = direction.astype(np.uint8)

# Resize images
scale = 0.5
fresh_img = cv2.resize(fresh_img, (0, 0), fx=scale, fy=scale)
x_edges = cv2.resize(x_edges, (0, 0), fx=scale, fy=scale)
y_edges = cv2.resize(y_edges, (0, 0), fx=scale, fy=scale)
result = cv2.resize(result, (0, 0), fx=scale, fy=scale)
direction = cv2.resize(direction, (0, 0), fx=scale, fy=scale)

# Display
cv2.imshow('RAW', fresh_img)
cv2.imshow('EdgesX', x_edges)
cv2.imshow('EdgesY', y_edges)
cv2.imshow('Result', result)
cv2.imshow('Direction', direction)
cv2.waitKey(0)
cv2.destroyAllWindows()
