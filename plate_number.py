import cv2 as cv
import numpy as np

# 读取图片
img_path = '10.jpg'
img = cv.imread(img_path)

cv.imshow('Original image', img)

cv.waitKey(0)
cv.destroyAllWindows()