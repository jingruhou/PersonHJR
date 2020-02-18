# coding: utf-8
import cv2
import numpy as np
import os

"""
    @Time    : 2020/2/17/0017 14:59
    @Author  : houjingru@semptian.com
    @FileName: frame2video.py
    @Software: PyCharm
"""
path = 'C:/Users/user/PycharmProjects/PersonHJR/Resource/Imgs'
filelist = os.listdir(path)

fps = 4  # 视频每秒25帧
size = (640, 480)  # 需要转化为视频的图片的尺寸
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter("C:/Users/user/PycharmProjects/PersonHJR/Resource/Video/result.avi", fourcc, fps, size)
#  视频保存在当前目录下
for item in filelist:
    if item.endswith('*.jpg'):
        item = path + item
        # 路径为中文名
        img = cv2.imdecode(np.fromfile(item, dtype=np.uint8), 1)
        # 路径为英文名
        img = cv2.imread(item)
        video.write(img)
video.release()
cv2.destroyAllWindows()
