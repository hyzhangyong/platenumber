import cv2 as cv
import numpy as np

# set blue thresh
lower_blue = np.array([200, 0, 0])
upper_blue = np.array([255, 100, 100])

# get a img and show
img = cv.imread('10.jpg')
cv.imshow('Img', img)

# get mask
mask = cv.inRange(img, lower_blue, upper_blue)
cv.imshow('Mask', mask)

cv.waitKey(0) & 0xFF
cv.destroyAllWindows()
