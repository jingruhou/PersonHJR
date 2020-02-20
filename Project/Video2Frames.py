# -*- coding: utf-8 -*-
import cv2
import os
import sys

"""
@Time    : 2020/2/10/0010 21:13
@Author  : houjingru@semptian.com
@FileName: Video2Frames.py
@Software: PyCharm
"""


def get_frame_timestamp(VideoCaptureObject):
    """
    获取帧戳
    :param VideoCaptureObject: opencv 的VideoCapture对象
    :return: frame_timestamp [时_分_秒_毫秒]
    """
    # 获取帧时间戳 2020-02-19 houjingru@semptian.com
    milliseconds = VideoCaptureObject.get(cv2.CAP_PROP_POS_MSEC)

    seconds = milliseconds // 1000
    milliseconds = milliseconds % 1000
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    time_stamp = [int(hours), int(minutes), int(seconds), int(milliseconds)]
    return time_stamp
    # return int(hours), int(minutes), int(seconds), int(milliseconds)
    # print("%s时%s分%s秒%s毫秒" % (int(hours), int(minutes), int(seconds), int(milliseconds)))


# 第一个输入参数是包含视频片段的路径
input_path = sys.argv[1]
# 第二个输入参数是设定每隔多少帧截取一帧
frame_interval = int(sys.argv[2])
# 列出文件夹下所有的视频文件
file_names = os.listdir(input_path)
print(file_names)
# 获取文件夹名称
video_prefix = input_path.split(os.sep)[-1]  # 获取视频文件夹名称 （Video）
# 建立一个新的文件夹，名称为原文件夹名称后加上_frames （Video_frames）
frame_path = '{}_frames'.format(input_path)
if not os.path.exists(frame_path):
    os.mkdir(frame_path)
# 初始化一个VideoCapture对象
cap = cv2.VideoCapture()

# 遍历所有文件
for file_name in file_names:
    file_path = os.sep.join([input_path, file_name])
    # VideoCapture::open函数可以从文件获取视频
    cap.open(file_path)
    # 获取视频帧数
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 同样为了避免视频头几帧质量低下，黑屏或者无关等
    for i in range(42):
        cap.read()
    for i in range(n_frames):
        ret, frame = cap.read()

        # 跳出判断
        if ret is False or frame is None:
            break
        # 每隔frame_interval帧进行一次截屏操作
        if i % frame_interval == 0:

            # # 获取帧时间戳 2020-02-19 houjingru@semptian.com
            #             # milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            #             # seconds = milliseconds // 1000
            #             # milliseconds = milliseconds % 1000
            #             # minutes = 0
            #             # hours = 0
            #             # if seconds >= 60:
            #             #     minutes = seconds // 60
            #             #     seconds = seconds % 60
            #             #
            #             # if minutes >= 60:
            #             #     hours = minutes // 60
            #             #     minutes = minutes % 60
            #             # print("%s时%s分%s秒%s毫秒" % (int(hours), int(minutes), int(seconds), int(milliseconds)))
            frame_time_stamp = get_frame_timestamp(cap)
            print(frame_time_stamp)

            image_name = '{}_{}_{:0>6d}_{}_{}_{}_{}.jpg'.format(video_prefix, file_name.split('.')[0], i, frame_time_stamp[0], frame_time_stamp[1], frame_time_stamp[2], frame_time_stamp[3])
            image_path = os.sep.join([frame_path, image_name])
            print('exported {}'.format(image_path))
            cv2.imwrite(image_path, frame)
# 执行结束释放资源
cap.release()
