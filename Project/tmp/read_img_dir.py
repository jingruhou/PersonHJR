# coding: utf-8
import cv2
import os
"""
    @Time    : 2020/2/18/0018 15:29
    @Author  : houjingru@semptian.com
    @FileName: read_img_dir.py
    @Software: PyCharm
"""
x = 0
for root, dirs, files in os.walk("C:/Users/user/PycharmProjects/PersonHJR/Resource/Video_frames"):
    for d in dirs:
        print(d)  # 打印子文件夹的个数
    for file in files:
        # 读入图像
        img_path = root+"/"+file
        img = cv2.imread(img_path, 1)


