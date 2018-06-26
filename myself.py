
# -*- coding: utf-8 -*-
"""
OpenCV中的霍夫变换
1.cv2.HoughLines(),返回一个(长度，角度)的数组，前者以像素为单位进行测量，后者以弧度为单位进行测量
2.该函数的第一个参数是二值图像，应用阈值或使用canny边缘检测，然后才应用霍夫变换。
  第二个参数和第三个参数分别为长度和角度，第四个参数是阈值。
"""

import cv2 as cv
import numpy as np

img = cv.imread('10.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
cv.imshow("hello", edges)

lines = cv.HoughLines(edges, 1, np.pi / 180, 150)
print('lines', lines, 'len', len(lines))

# 获取霍夫线数组长度
for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        print("%d, %d\n", x1, y1)
        print("%d, %d\n", x2, y2)

        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imwrite('houghlines3.png', img)
cv.imshow('res', img)
cv.waitKey(0) & 0xFF
cv.destroyAllWindows()

