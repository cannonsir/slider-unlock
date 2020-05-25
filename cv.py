import numpy as np
import cv2

import numpy as np
import random
import math

import requests
from selenium.webdriver import ActionChains
import time
from selenium import webdriver
from PIL import Image
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def external_rect_cut(image):
    """
    按照最边缘轮廓剪切出主体部分(矩形)
    :param image: 图片Mat
    :return:
    """

    # 颜色转换为灰度图像
    temp_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 阀值处理
    ret, thresh = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY)

    # 寻找轮廓,注意这里第二个参数表示只查询最外层的轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 获取第一个轮廓
    cnt = contours[-1]

    # 除轮廓外的区域做纯色处理
    for row in range(len(image)):
        for col in range(len(image[row])):
            # 不知道这个地方是为行和列弄反了还是怎么的，这里要反着传递给pointPolygonTest才正常
            # pointPolygonTest函数功能为检测某pixel是否在轮廓内，当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零.
            if cv2.pointPolygonTest(cnt, (col, row), False) < 0:
                # 文档说用NumPy是用于快速数组计算的优化库，简单地访问每个像素值并对其进行修改将非常缓慢？
                # image[row, col] = [0, 0, 0]

                # 这里使用文档推荐的方式来修改pixel
                image.itemset((row, col, 0), 0)
                image.itemset((row, col, 1), 0)
                image.itemset((row, col, 2), 0)

    # 展示轮廓效果
    # show(cv2.drawContours(image, cnt, -1, (0, 255, 0), 1))

    # 极点是指对象的最顶部，最底部，最右侧和最左侧的点, 返回值为 x，y
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # 绘制极点,展示极点绘制结果
    # cv2.circle(image, leftmost, 1, (255, 0, 0), 2)
    # cv2.circle(image, rightmost, 1, (0, 255, 0), 2)
    # cv2.circle(image, topmost, 1, (0, 0, 255), 2)
    # cv2.circle(image, bottommost, 1, (255, 255, 0), 2)
    # show(image)

    # 裁切
    temp_img = image[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

    # 写入日志
    cv2.imwrite("logs/cut.png", temp_img)

    # 返回裁剪结果
    return temp_img


# opencv2不规则裁剪
def ROI_byMouse(img, pts):
    mask = np.zeros(img.shape, np.uint8)
    # pts = np.array(lsPointsChoose, np.int32)  # pts是多边形的顶点列表（顶点集）
    col0 = pts[:, 0]
    col1 = pts[:, 1]
    x1 = np.min(col0)
    y1 = np.min(col1)
    x2 = np.max(col0)
    y2 = np.max(col1)
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制

    # --------------画多边形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    # -------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)

    return ROI[y1:y2, x1:x2]


def searchBySide(template, contour, debug=False):
    gary = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 这里阀值保留220以上的白色区域，因为滑块目标有明显的白色轮廓
    ret, gary = cv2.threshold(gary, 220, 255, cv2.THRESH_BINARY)

    # 宽高
    x, y, w, h = cv2.boundingRect(contour)

    if debug:
        x = cv2.Sobel(gary, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gary, cv2.CV_16S, 0, 1)
        # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
        # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        cv2.imshow('1', gary)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    blur = cv2.GaussianBlur(gary, (5, 5), 0)  # 用高斯滤波处理原图像降噪
    canny = cv2.Canny(blur, 0, 150)  # 50是最小阈值,150是最大阈值

    debug and show(canny)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    minRet = 5.00
    position = None

    for index in range(len(contours)):
        cnt = contours[index]
        ret = cv2.matchShapes(contour, cnt, 1, 0.0)

        # 打印调试匹配的相似度
        # print(ret)
        # show(cv2.drawContours(template.copy(), cnt, -1, (0, 255, 0), 3))

        # 求宽高比
        x, y, sub_w, sub_h = cv2.boundingRect(cnt)

        # 对相似度和长宽比做比较
        if ret < minRet and (sub_w + 5) > w > (sub_w - 5) and (sub_h + 5) > h and (sub_h - 5) < h:

            minRet = ret

            # 轮廓极端点检测, x,y
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])

            # 取最左侧的x和最顶部的Y即为左上角定位
            position = (leftmost[0], topmost[1])

    return position


# TODO 目前这里有问题，如果是凸滑块可能找不到,应该用candy+轮廓检测
def global_match(template, target):
    """
    全局模版匹配
    :param template: 模版图片路径
    :param target: 缺口图片路径
    :return: 位置 x, y
    """

    a = template

    # 读取图片
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(template, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for index in range(len(contours)):
        show(cv2.drawContours(a.copy(), contours[index], -1, (0, 255, 0), 2))

    # 获取缺口图片宽高
    w, h = target.shape[::-1]

    # 指定副本文件存储位置
    temp = 'logs/temp.jpg'
    targ = 'logs/targ.jpg'

    # 复制图片为副本
    cv2.imwrite(temp, template)
    cv2.imwrite(targ, target)

    # 读取图片为MAT
    target = cv2.imread(targ)
    # 转换颜色为灰色
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # 递归numpy数组中的每一项，对每一项进行减255的运算，然后求绝对值(取对比色?)
    target = abs(255 - target)

    cv2.imwrite(targ, target)

    # 二次读取
    target = cv2.imread(targ)
    template = cv2.imread(temp)

    # 调用openCV模版匹配获取匹配结果
    result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF)

    # 获取坐标信息
    y, x = np.unravel_index(result.argmax(), result.shape)

    # 展示圈出来的区域
    cv2.rectangle(template, (x, y), (x + w, y + h), (7, 249, 151), 2)
    cv2.imwrite("logs/matchTemplateRes.jpg", template)

    # 弹窗展示
    # show(template)

    # 返回左上角坐标信息
    return x, y


def show(name):
    cv2.imshow('Show', name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def urllib_download(imgurl, imgsavepath):
    """
    下载图片
    :param imgurl: 图片url
    :param imgsavepath: 存放地址
    :return:
    """
    from urllib.request import urlretrieve
    urlretrieve(imgurl, imgsavepath)


def find_outline_position(template, target):
    """
    在指定模版中通过指定目标的轮廓查找在模版中的位置
    :param template:
    :param target:
    :return:
    """

    # 颜色转换
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(target, 65, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    targetCnt = contours[0]

    # 这里阀值要控制的很好，不然找不到
    ret, thresh = cv2.threshold(template, 86, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # show(thresh)
    cv2.imwrite('./logs/find_outline_position_thresh.png', thresh)

    minRet = 1
    minPosition = None

    for index in range(len(contours)):
        ret = cv2.matchShapes(targetCnt, contours[index], 1, 0.0)

        # 打印调试匹配的相似度
        # print(ret)
        # show(cv2.drawContours(horizon_bar.copy(), contours[index], -1, (0, 255, 0), 1))

        if ret < minRet:
            minRet = ret
            # 寻找轮廓的最左侧点
            minPosition = tuple(contours[index][contours[index][:, :, 0].argmin()][0])

    return minPosition


def fetch():
    # selenium选项
    options = webdriver.ChromeOptions()

    # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
    options.add_experimental_option('excludeSwitches', ['enable-automation'])

    # 创建会话
    session = webdriver.Remote(command_executor='http://127.0.0.1:4444/wd/hub', options=options)

    # 窗口最大话
    session.maximize_window()

    session.get('https://ibaotu.com/')

    # QQ登录
    ActionChains(session) \
        .click(session.find_element_by_xpath("(//a[contains(text(),'请登录')])[2]")) \
        .click(session.find_element_by_xpath("//a[2]/p/i")) \
        .perform()

    # 切换到登录交互框的上下文中
    session.switch_to.frame("ptlogin_iframe")

    ActionChains(session) \
        .click(session.find_element_by_id("switcher_plogin")) \
        .send_keys_to_element(session.find_element_by_id("u"), "1234567888") \
        .send_keys_to_element(session.find_element_by_id("p"), "xxx") \
        .click(session.find_element_by_xpath("//div[@class='submit']/a")) \
        .perform()

    # 等待验证码iframe
    WebDriverWait(session, 10).until(lambda x: x.find_element_by_id("newVcodeArea"))

    time.sleep(1)

    # 切换到验证码 TODO 有时候没有验证码的情况不需要以下步骤了
    session.switch_to.frame("tcaptcha_iframe")

    time.sleep(2)

    # 等待验证码图片,并截图保存
    WebDriverWait(session, 10).until(lambda x: x.find_element_by_id("slideBg")).screenshot('./img/intersection.png')

    block = session.find_element_by_xpath('//img[@id="slideBg"]').get_attribute('src')  # 大图 url
    sub = session.find_element_by_xpath('//img[@id="slideBlock"]').get_attribute('src')  # 小滑块 图片url

    # 保存图片至本地
    urllib_download(block, './img/template.png')
    urllib_download(sub, './img/slider.png')

    time.sleep(5)

    session.quit()


def fetchTest():
    # selenium选项
    options = webdriver.ChromeOptions()

    # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
    options.add_experimental_option('excludeSwitches', ['enable-automation'])

    # 创建会话
    driver = webdriver.Remote(command_executor='http://127.0.0.1:4444/wd/hub', options=options)
    driver.maximize_window()
    driver.get("https://open.captcha.qq.com/online.html")

    click_keyi_username = driver.find_element_by_xpath("//div[@class='wp-onb-tit']/a[text()='可疑用户']")
    WebDriverWait(driver, 10, 5).until(lambda dr: click_keyi_username).click()

    login_button = driver.find_element_by_id('code')
    WebDriverWait(driver, 10, 5).until(lambda dr: login_button).click()
    time.sleep(1)

    driver.switch_to.frame(driver.find_element_by_id('tcaptcha_iframe'))  # switch 到 滑块frame
    time.sleep(0.5)
    bk_block = driver.find_element_by_xpath('//img[@id="slideBg"]')  # 大图
    web_image_width = bk_block.size
    web_image_width = web_image_width['width']
    bk_block_x = bk_block.location['x']

    slide_block = driver.find_element_by_xpath('//img[@id="slideBlock"]')  # 小滑块
    slide_block_x = slide_block.location['x']

    for index in range(5):
        print('当前刷新第' + str(index) + '次验证码')

        bk_block = driver.find_element_by_xpath('//img[@id="slideBg"]').get_attribute('src')  # 大图 url
        slide_block = driver.find_element_by_xpath('//img[@id="slideBlock"]').get_attribute('src')  # 小滑块 图片url
        slid_ing = driver.find_element_by_xpath('//div[@id="tcaptcha_drag_thumb"]')  # 滑块

        urllib_download(bk_block, './img/template.png')
        urllib_download(slide_block, './img/slider.png')

        # 读取图片
        template = cv2.imread('img/template.png')
        slider = cv2.imread('img/slider.png')

        bigCut = external_rect_cut(slider)
        bigCutCnt, t = cv2.findContours(cv2.cvtColor(bigCut, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)

        res = searchBySide(template, bigCutCnt[0])

        draw = cv2.drawMarker(template.copy(), res, (0, 0, 255), markerSize=70, thickness=3)
        filename = 'test/res_' + str(int(time.time())) + '_template.png'
        cv2.imwrite(filename, template)
        cv2.imwrite('test/res_' + str(int(time.time())) + '_res.png', draw)
        cv2.imwrite('test/res_' + str(int(time.time())) + '_slider.png', slider)

        cv2.imshow(filename, draw)

        ActionChains(driver) \
            .click(driver.find_element_by_id("reload")) \
            .perform()

        time.sleep(1)

    cv2.waitKey(0)
    driver.quit()
    quit()


# TODO 以下为方案A，如果不行，就用global_match
# TODO 1. 剪切intersection左边一半内容来查找target，找到初始时的target的bbox值
# TODO 2. 取bbox的上下x坐标及最右侧y坐标，剪切intersection为一个包含最终目标点的横条
# TODO 3. 通过模版匹配在横条中搜索target轮廓，确定目标bbox，返回目标bbox的左侧y值

# 爬取网页
# fetch()

fetchTest()
quit()

# 读取图片
# intersection = cv2.imread('img/intersection.png')
template = cv2.imread('test/res_1587387491_template.png')
slider = cv2.imread('test/res_1587387491_slider.png')

bigCut = external_rect_cut(slider)
bigCutCnt, t = cv2.findContours(cv2.cvtColor(bigCut, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

res = searchBySide(template, bigCutCnt[0])

cv2.drawMarker(template, res, (0, 0, 255), markerSize=70, thickness=3)

print('滑块最终坐标点:')
print(res)

cv2.imwrite('logs/res_' + str(int(time.time())) + '.png', template)

show(template)

quit()

# img = cv2.imread('img/intersection.png', 0)#转化为灰度图
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
# Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
# Scale_absY = cv2.convertScaleAbs(y)
# result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
# cv2.imshow('1', img)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.imread('img/intersection.png', 0)#转化为灰度图
# img_color = img
# blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
# canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
# cv2.namedWindow("canny",0);#可调大小
# cv2.namedWindow("1",0);#可调大小
# cv2.imshow('1', img)
# cv2.imshow('canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


bigCut = external_rect_cut(slider)
bigCutCnt, t = cv2.findContours(cv2.cvtColor(bigCut, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 获取目标滑块与截图之间的比例差距
iShape = intersection.shape
tShape = template.shape
sShape = slider.shape

# 由于template和target是一套的,图片比例相同,所以intersection(截图)和template(模版)的比例就是截图中滑块与target滑块的比例
# 这里统一3张图片的大小比例
if iShape[0] > tShape[0]:
    scale = iShape[0] / tShape[0]
    intersection = cv2.resize(intersection, (int(iShape[1] / scale, int(iShape[0] / scale))),
                              interpolation=cv2.INTER_CUBIC)
# 截图区域小于等于模版
else:
    scale = tShape[0] / iShape[0]
    template = cv2.resize(template, (int(tShape[1] / scale), int(tShape[0] / scale)), interpolation=cv2.INTER_CUBIC)
    slider = cv2.resize(slider, (int(sShape[1] / scale), int(sShape[0] / scale)), interpolation=cv2.INTER_CUBIC)

# 调用提取主体部分函数，获取目标方块的干净图
cut = external_rect_cut(slider)

# 查找截图的左侧部分，得到初始滑块的左上角坐标(因为针对腾讯防水墙业务，左侧2/1区域必然包含滑块初始位置)
start_x = int(intersection.shape[1] * (30 / 280) - 2)

left_top_point = searchBySide(intersection[:, start_x:start_x + cut.shape[1] + 10], bigCutCnt[0])
left_top_point = (left_top_point[0] + start_x, left_top_point[1])

# 通过滑块在截图中左上角的位置求出坐下角的坐标
left_bottom_point = (left_top_point[0], left_top_point[1] + cut.shape[0])

# 保存日志绘制
init_position_draw = intersection
cv2.drawMarker(init_position_draw, left_top_point, (0, 255, 0))
cv2.drawMarker(init_position_draw, left_bottom_point, (0, 255, 0))
cv2.imwrite('./logs/init_position_draw.png', init_position_draw)

# 剪切出右半边的滑动区域的横条，方便最终模版查询精度提升
y_start = left_top_point[1] - 2
horizon_bar = intersection[y_start:left_bottom_point[1] + 5, math.ceil(intersection.shape[1] / 2):]
cv2.imwrite('./logs/horizon_bar.png', horizon_bar)

target_position = searchBySide(horizon_bar, bigCutCnt[0], True)

if target_position == None:
    quit('没找到')

show(cv2.drawMarker(intersection, (int(intersection.shape[1] / 2) + target_position[0], target_position[1] + y_start),
                    (0, 255, 0)))

# 已成功图案：沙漠，企鹅,岩壁,公路
