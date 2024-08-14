import cv2
import numpy as np

img = cv2.imread('Assets/cat.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (0,0), fx=10, fy=10)
result = np.ones(img.shape)
img = cv2.GaussianBlur(img, (15,15), 0)


def greatest_difference_within_distance(array, coords, distance):
    x, y = coords
    origin = array[y][x]
    
    x_start = max(0, x - distance)
    x_end = min(array.shape[1], x + distance + 1)
    y_start = max(0, y - distance)
    y_end = min(array.shape[0], y + distance + 1)

    values = np.array(array[y_start:y_end, x_start:x_end])

    brightestDiff = abs(np.subtract(origin, values.max()))
    
    return brightestDiff
    # return np.average(values) / values.size()

for y in range(0, len(img)):
    for x in range(0,len(img[y])):
        # print(img[j][i])
        diff = greatest_difference_within_distance(img, (x, y), 1)
        # result[y][x] = diff
        if (diff > 254):
            result[y][x] = 0
        else:
            result[y][x] = 1
    # print(diff)

scale = .5
img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
result = cv2.resize(result, (0, 0), fx=scale, fy=scale)
cv2.imshow('RAW', img)
cv2.waitKey(0)
cv2.imshow('Edges', result)
cv2.waitKey(0)
cv2.destroyAllWindows()