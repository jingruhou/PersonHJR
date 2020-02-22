# -*- coding: utf-8 -*-
import cv2
"""
    @Time    : 2020/2/12/0012 11:12
    @Author  : houjingru@semptian.com
    @FileName: getFrameTimestamp.py
    @Software: PyCharm
"""
cap = cv2.VideoCapture('C:/Users/user/PycharmProjects/PersonHJR/Resource/Video/sorted_sim.mp4')
success, frame = cap.read()

while success:
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow('FrameTimestamp', frame)
    success, frame = cap.read()
    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

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

    print("%s时%s分%s秒%s毫秒" % (int(hours), int(minutes), int(seconds), int(milliseconds)))
cv2.destroyAllWindows()
cap.release()