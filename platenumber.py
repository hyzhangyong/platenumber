import cv2 as cv
import numpy as np
from aip import AipOcr


def get_plate_number(img_path):
    img = cv.imread(img_path)

    # 通过hsl找蓝色
    lower_blue = np.array([0, 20, 200])
    upper_blue = np.array([150, 150, 255])
    hsl = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    mark = cv.inRange(hsl, lower_blue, upper_blue)

    # 形态学变换
    gaussian = cv.GaussianBlur(mark, (3, 3), 0, 0, cv.BORDER_DEFAULT)  # 高斯模糊
    median = cv.medianBlur(gaussian, 5)  # 中值滤波
    ret, binary = cv.threshold(median, 170, 255, cv.THRESH_BINARY)  # 二值化

    # 做膨胀与腐蚀操作
    element1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    element2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 7))

    dilation = cv.dilate(binary, element2, iterations=1)  # 膨胀一次，让轮廓突出
    erosion = cv.erode(dilation, element1, iterations=1)  # 腐蚀一次，去掉细节
    dilation2 = cv.dilate(erosion, element2, iterations=3)  # 再次膨胀，让轮廓明显一些

    # 查找轮廓
    img2, contours, hierarchy = cv.findContours(dilation2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    region = []

    # 筛选面积小的
    for i in range(len(contours)):
        cnt = contours[i]

        # 计算该轮廓的面积
        area = cv.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 2000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv.minAreaRect(cnt)

        # box是四个点的坐标
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 车牌正常情况下长高比在2.7-5之间
        ratio = float(width) / float(height)

        if ratio > 5 or ratio < 2:
            continue
        region.append(box)

    # 用绿线画出这些找到的轮廓
    for box in region:
        cv.drawContours(img, [box], 0, (0, 255, 0), 2)
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        img_org2 = img.copy()
        img_plate = img_org2[y1:y2, x1:x2]

        cv.imwrite("temp.jpg", img_plate)

        # 利用百度AI进行车牌的字符识别
        APP_ID = '11474009'
        API_KEY = '4PbKqW6ADkz6pG56hg4rxGiw'
        SECRET_KEY = 'zqvGGVUfXrNV4CttvnXx4EwgESrG5Mdv'
        client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

        with open('temp.jpg', 'rb') as fp:
            img = fp.read()

        result = client.licensePlate(img)
        return result['words_result']['number']
