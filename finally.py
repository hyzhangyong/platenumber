from aip import AipOcr


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 利用百度AI进行车牌的字符识别
def BaiduAI(filePath):
    APP_ID = '11469861'
    API_KEY = '2NmBChTTNG33BVixnbiZiMLR'
    SECRET_KEY = 'l2gjFmGcxgnUUw6WIBQIGUVQkEqqnGNM'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    img = get_file_content(filePath)
    result = client.licensePlate(img)
    return result


result = BaiduAI("number_plate2.jpg")
print(result['words_result']['number'])
