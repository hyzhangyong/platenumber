import cv2 as cv
import numpy as np

# set blue thresh
import pytesseract

lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 100, 100])

# get a img and show
img = cv.imread('number_plate.jpg')
cv.imshow('Img', img)

# get mask
mask = cv.inRange(img, lower_blue, upper_blue)
cv.imshow('Mask', mask)

cv.imwrite("temp.jpg", mask)


cv.waitKey(0) & 0xFF
cv.destroyAllWindows()
