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

