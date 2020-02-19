# -*- coding: utf-8 -*-
from __future__ import division
import mxnet as mx
import numpy as np
import cv2

"""
    @Time    : 2020/2/12/0014 12:06
    @Author  : houjingru@semptian.com
    @FileName: compute_sim.py
    @Software: PyCharm
"""


def read_face_data(file_path):
    """
    读取文件夹里面的文本文件
    :param file_path:
    :return: 文本内容
    """
    with open(file_path, "r") as f:
        data = f.read().splitlines()
        return data


def split_face_data(data):
    """
    根据相关分隔符过滤相关信息
    :param data:人脸元数据/文本内容
    :return:人脸特征
    """
    # 检索文件内容“normed_embedding”检索人脸正则化后的特征
    # face_normed_embedding = data


face0 = read_face_data("./faces_metadata/Video_1_000000_face0.txt")

face1 = read_face_data("./faces_metadata/Video_1_000025_face0.txt")



def compute_sim(self, img1, img2):
    """
        计算余弦相似度
    :param self:
    :param img1: 图片1
    :param img2: 图片2
    :return: 相似度
    """
    emb1 = self.get_embedding(img1).flatten()
    emb2 = self.get_embedding(img2).flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim
