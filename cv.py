import numpy as np
import cv2 as cv

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


def external_rect_cut(image):
    """
    按照最边缘轮廓剪切出主体部分(矩形)
    :param image: 图片Mat
    :return:
    """
    temp_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #
    ret, thresh = cv.threshold(temp_img, 127, 255, 0)

    # 寻找轮廓,注意这里第二个参数表示只查询最外层的轮廓
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 获取第一个轮廓
    cnt = contours[0]

    # 极点是指对象的最顶部，最底部，最右侧和最左侧的点
    (l_x, l_y) = tuple(cnt[cnt[:, :, 0].argmin()][0])
    (r_x, r_y) = tuple(cnt[cnt[:, :, 0].argmax()][0])
    (t_x, t_y) = tuple(cnt[cnt[:, :, 1].argmin()][0])
    (b_x, b_y) = tuple(cnt[cnt[:, :, 1].argmax()][0])

    # 绘制极点,展示极点绘制结果
    # test_show = target
    # cv.circle(test_show, (l_x, l_y), 1, (255, 0, 0), 3)
    # cv.circle(test_show, (r_x, r_y), 1, (0, 255, 0), 3)
    # cv.circle(test_show, (t_x, t_y), 1, (0, 0, 255), 3)
    # cv.circle(test_show, (b_x, b_y), 1, (255, 255, 0), 3)
    # cv.imshow('points', test_show)

    # 裁切并返回裁切结果
    return image[l_x:r_x, t_y:b_y]


def global_match(template, target):
    """
    全局模版匹配
    :param template: 模版图片路径
    :param target: 缺口图片路径
    :return: 位置 x, y
    """

    # 读取图片
    template = cv.imread(template, 0)
    target = cv.imread(target, 0)

    print(target.dtype)

    # 获取缺口图片宽高
    w, h = target.shape[::-1]

    # 指定副本文件存储位置
    temp = 'logs/temp.jpg'
    targ = 'logs/targ.jpg'

    # 复制图片为副本
    cv.imwrite(temp, template)
    cv.imwrite(targ, target)

    # 读取图片为MAT
    target = cv.imread(targ)
    # 转换颜色为灰色
    target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    # 递归numpy数组中的每一项，对每一项进行减255的运算，然后求绝对值(取对比色?)
    target = abs(255 - target)

    # 保存以上处理步骤的结果为图片文件
    cv.imwrite(targ, target)

    # 二次读取
    target = cv.imread(targ)
    template = cv.imread(temp)

    # 调用openCV模版匹配获取匹配结果
    result = cv.matchTemplate(target, template, cv.TM_CCOEFF_NORMED)

    print(result.argmax())

    print(323)

    # 获取坐标信息
    x, y = np.unravel_index(result.argmax(), result.shape)

    # 展示圈出来的区域
    cv.rectangle(template, (y, x), (y + w, x + h), (7, 249, 151), 2)
    cv.imwrite("logs/yuantu.jpg", template)

    # 弹窗展示
    show(template)

    # 返回坐标信息
    return x, y


def show(name):
    cv.imshow('Show', name)
    cv.waitKey(0)
    cv.destroyAllWindows()


def urllib_download(imgurl, imgsavepath):
    """
    下载图片
    :param imgurl: 图片url
    :param imgsavepath: 存放地址
    :return:
    """
    from urllib.request import urlretrieve
    urlretrieve(imgurl, imgsavepath)


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
urllib_download(sub, './img/target.png')

time.sleep(5)

session.quit()

# cv 操作

# TODO 以下为方案A，如果不行，就用global_match
# TODO 1. 剪切intersection左边一半内容来查找target，找到初始时的target的bbox值
# TODO 2. 取bbox的上下x坐标及最右侧y坐标，剪切intersection为一个包含最终目标点的横条
# TODO 3. 通过模版匹配在横条中搜索target轮廓，确定目标bbox，返回目标bbox的左侧y值

intersection = cv.imread('img/intersection.png')
template = cv.imread('img/template.png')
target = cv.imread('img/target.png')

# get_init_bbox(intersection, target)
# position = globalMatch('./img/template.png', './img/target.png')
# print(position)

cut = external_rect_cut(target)
cv.imshow('', cut)
cv.waitKey(0)
