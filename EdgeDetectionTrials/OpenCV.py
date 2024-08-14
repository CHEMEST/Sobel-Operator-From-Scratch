import cv2
import numpy as np
from scipy.ndimage import convolve

input_img = cv2.imread('Assets/itachi.jpg', cv2.IMREAD_GRAYSCALE)

# result = cv2.Laplacian(input_img, ddepth=cv2.CV_16S, ksize=3)
result = cv2.GaussianBlur(input_img , (15, 15), 0)

result = cv2.Sobel(input_img,cv2.CV_64F,0,1,ksize=5)

scale = .5
input_img = cv2.resize(input_img, (0, 0), fx=scale, fy=scale)
result = cv2.resize(result, (0, 0), fx=scale, fy=scale)


cv2.imshow('RAW', input_img)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()