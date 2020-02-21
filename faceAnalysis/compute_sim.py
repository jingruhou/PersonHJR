# -*- coding: utf-8 -*-
import json
import numpy as np

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
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        return data


def split_face_data(face_data):
    """
    根据相关分隔符过滤相关信息
    :param face_data:人脸元数据/文本内容
    :return:face_meta_data list
    """
    face_tmp1 = face_data['embedding'].strip('[]')
    face_tmp2 = face_tmp1.split()
    return face_tmp2


def compute_sim(embedding1, embedding2):
    """
        计算余弦相似度
    :param embedding1: 图片1
    :param embedding2: 图片2
    :return: 相似度
    """
    from numpy.linalg import norm
    sim_emb = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return sim_emb


########################################################################################################################
face0 = read_face_data("./faces_metadata/Video_1_000000_0_0_1_680_face0.json")
face0_embedding = split_face_data(face0)
# face0_embedding格式为list，其中每一个value为str，需要转化为float32
face0_embedding_float = []
for index, value in enumerate(face0_embedding):
    item_float = float(value)
    face0_embedding_float.append(item_float)
print(face0_embedding_float)

########################################################################################################################
face1 = read_face_data("./faces_metadata/Video_1_000025_0_0_2_680_face0.json")
face1_embedding = split_face_data(face1)
# face1_embedding格式为list，其中每一个value为str，需要转化为float32
face1_embedding_float = []
for index, value in enumerate(face1_embedding):
    item_float = float(value)
    face1_embedding_float.append(item_float)
print(face1_embedding_float)

# 计算余弦相似度
sim_embedding = compute_sim(face0_embedding_float, face1_embedding_float)
print("相似度为：%s" % sim_embedding)
