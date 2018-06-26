import cv2 as cv
import numpy as np

# 读取图片
img_path = '10.jpg'
img = cv.imread(img_path)
cv.imshow('Original image', img)

# 转化成灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray image', gray)

# 形态学变换
gaussian = cv.GaussianBlur(gray, (3, 3), 0, 0, cv.BORDER_DEFAULT)
# cv.imshow('Gaussian image', gaussian)

# 中值滤波
median = cv.medianBlur(gaussian, 5)
# cv.imshow('Median image', median)

# Sobel算子， X方向求梯度
sobel = cv.Sobel(median, cv.CV_8U, 1, 0, ksize=3)
# cv.imshow('Sobel image', sobel)

# 二值化
ret, binary = cv.threshold(sobel, 170, 255, cv.THRESH_BINARY)
# cv.imshow('Binary image', binary)

# 膨胀和腐蚀操作的核函数
element1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
element2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 7))

# 膨胀一次，让轮廓突出
dilation = cv.dilate(binary, element2, iterations=1)
# cv.imshow('Dilation image', dilation)

# 腐蚀一次，去掉细节
erosion = cv.erode(dilation, element1, iterations=1)
cv.imshow('Erosion image', erosion)

# 再次膨胀，让轮廓明显一些
dilation2 = cv.dilate(erosion, element2, iterations=3)
cv.imshow('Dilation2 image', dilation2)

# 查找轮廓
contours = cv.findContours(dilation2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours)

cv.waitKey(0)
cv.destroyAllWindows()
