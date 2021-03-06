# -*- coding: utf-8 -*-

"""
    @Time    : 2020/2/10/0014 12:04
    @Author  : houjingru@semptian.com
    @FileName: Deduplication.py
    @Software: PyCharm
"""

"""
    sorted_sim、帧内去重（所有帧）- 略(忽略这种场景)
    
    计算每一帧内的人脸feature的相似度，过滤掉同一个人 - 确保一人一个feature
"""

"""
    2、帧间去重（所有帧）
    
    计算两帧之间的所有人脸feature的相似度，过滤掉同一个人 - 确保一人一个feature
"""

"""
    3、单个视频内人物统计
    
    某个视频内人物/人脸去重，最终得到是这个影片中所有的人脸feature-入库
    记录每一个face出现过的帧戳和次数，进行次数统计，并且排名
    
    
"""

"""
    4、多个视频内人物统计
    
    多个视频内人物/人脸去重，合并-更新库
    记录每一个face出现过的帧戳和次数，进行全库次数统计，并且全库排名
"""

