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


def urllib_download(imgurl, imgsavepath):
    """
    下载图片
    :param imgurl: 图片url
    :param imgsavepath: 存放地址
    :return:
    """
    from urllib.request import urlretrieve
    urlretrieve(imgurl, imgsavepath)

# 获取方块初始bbox
def get_init_bbox(all, sub):

    h, w, s = all.shape

    # 使用numpy数组切片对图像进行剪裁
    cropped = cv.cvtColor(all[:, :math.ceil(w / 3)], cv.COLOR_BGR2GRAY)

    sub = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)

    # cv.imshow('cropped', cropped)
    # cv.imshow('sub', sub)
    # cv.threshold(sub, sub, 254,255, cv.CV_THRESH_BINARY)

    # TODO 关于验证码的拖动块的留白区域删除，可以通过极端点算出bbox再画矩形

    # TODO 模版与目录大小不一致

    # 不知名群友回答
    # pyrDown*2
    # cvtColor
    # medianBlur
    # GaussianBlur
    # threshold
    # 一套下来
    # 再来边缘检测+获取凸包应该就可以了
    #

    res = cv.matchTemplate(cv.cvtColor(all, cv.COLOR_BGR2GRAY), sub, cv.TM_CCOEFF)

    x, y = np.unravel_index(res.argmax(), res.shape)

    cv.rectangle(all, (y, x), (y + w, x + h), (7, 249, 151), 2)
    cv.imwrite("yuantu.jpg", all)
    cv.imshow('aaa', all)
    print(res)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    left_top = max_loc  # 左上角
    right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
    cv.rectangle(cropped, left_top, right_bottom, 255, 2)  # 画出矩形位置
    cv.waitKey(0)

    return 233

# # selenium选项
# options = webdriver.ChromeOptions()
#
# # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
# options.add_experimental_option('excludeSwitches', ['enable-automation'])
#
# # 创建会话
# session = webdriver.Remote(command_executor='http://127.0.0.1:4444/wd/hub', options=options)
#
# # 窗口最大话
# session.maximize_window()
#
# session.get('https://ibaotu.com/')
#
# # QQ登录
# ActionChains(session) \
#     .click(session.find_element_by_xpath("(//a[contains(text(),'请登录')])[2]")) \
#     .click(session.find_element_by_xpath("//a[2]/p/i")) \
#     .perform()
#
# # 切换到登录交互框的上下文中
# session.switch_to.frame("ptlogin_iframe")
#
# ActionChains(session) \
#     .click(session.find_element_by_id("switcher_plogin")) \
#     .send_keys_to_element(session.find_element_by_id("u"), "1234567888") \
#     .send_keys_to_element(session.find_element_by_id("p"), "xxx") \
#     .click(session.find_element_by_xpath("//div[@class='submit']/a")) \
#     .perform()
#
# # 等待验证码iframe
# WebDriverWait(session, 10).until(lambda x: x.find_element_by_id("newVcodeArea"))
#
# # 切换到验证码
# session.switch_to.frame("tcaptcha_iframe")
#
# time.sleep(2)
#
# # 等待验证码图片
# WebDriverWait(session, 10).until(lambda x: x.find_element_by_id("slideBg")).screenshot('./all.png')
#
# block = session.find_element_by_xpath('//img[@id="slideBg"]').get_attribute('src')  # 大图 url
# sub = session.find_element_by_xpath('//img[@id="slideBlock"]').get_attribute('src')  # 小滑块 图片url
#
# # 保存图片至本地
# urllib_download(block, './block.png')
# urllib_download(sub, './sub.png')

# session.quit()

# cv 操作

cv_all = cv.imread('./all.png')
cv_block = cv.imread('./block.png')
cv_sub = cv.imread('./sub.png')

init_bbox = get_init_bbox(cv_all, cv_sub)

cv.waitKey(0)
