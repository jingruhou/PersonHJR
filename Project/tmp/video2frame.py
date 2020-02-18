# coding: utf-8
import cv2
"""
    @Time    : 2020/2/17/0017 15:20
    @Author  : houjingru@semptian.com
    @FileName: video2frame.py
    @Software: PyCharm
"""
path = '../Resource/1.mp4'
cap = cv2.VideoCapture(path)
suc = cap.isOpened()  # 是否成功打开
frame_count = 0
while suc:
    frame_count += 1
    suc, frame = cap.read()
    params = [2]
    cv2.imwrite('frames/%d.jpg' % frame_count, frame, params)
    # （判断帧数是否等于视频总帧数）跳出循环
    if frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        break

cap.release()
print('unlock movie: ', frame_count)
