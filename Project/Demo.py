# coding: utf-8

import cv2
import os
import sys

"""
    @Time    : 2020/2/10/0018 11:02
    @Author  : houjingru@semptian.com
    @FileName: Demo.py
    @Software: PyCharm
"""


def Video_fetching(input_path, frame_interval):
    """
    视频抽帧
    :param input_path:视频输入路径
    :param frame_interval:帧间隔
    :return:
    """
    filenames = os.listdir(input_path)  # 列出文件夹下所有的视频文件
    video_prefix = input_path.split(os.sep)[-1]  # 获取文件夹名称
    frame_path = '{}_frames'.format('video')  # 新建文件夹，名称为原名加上_frames
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    cap = cv2.VideoCapture()  # 初始化一个VideoCapture对象

    for filename in filenames:
        if filename.endswith('.mp4'):
            filepath = os.sep.join([input_path, filename])
            cap.open(filepath)  # VideoCapture::open函数可以从文件获取视频
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频帧数

            for i in range(42):  # 为了避免视频头几帧质量低下，黑屏或者无关等
                cap.read()
            for i in range(n_frames):
                ret, frame = cap.read()

                # 每隔frame_interval帧进行一次截屏操作
                if i % frame_interval == 0:
                    imagename = '{}_{}_{:0>6d}.jpg'.format(video_prefix, filename.split('.')[0], i)
                    imagepath = os.sep.join([frame_path, imagename])
                    print('exported {}'.format(imagepath))
                    cv2.imwrite(imagepath, frame)

        else:
            print('This file is not video')
    cap.release()  # 执行结束释放资源


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('依次传入参数：视频片段的路径(文件夹)，设定每隔多少帧截取一帧int(20)')
        exit(0)
    Video_fetching(sys.argv[1], int(sys.argv[2]))
