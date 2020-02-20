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
    # 检索文件内容“normed_embedding”检索人脸正则化后的特征
# out=str.replace('[','').replace(']','')

# face0 = read_face_data("./faces_metadata/Video_1_000000_0_0_1_680_face0.json")
# face0_embedding = np.array(face0['embedding'].replace('[', '').replace(']', '').split(" "))
# face0_normed_embedding = np.array(face0['normed_embedding'].replace('[', '').replace(']', '').split(" "))
#
# face1 = read_face_data("./faces_metadata/Video_1_000025_0_0_2_680_face0.json")
# face1_embedding = np.array(face1['embedding'].replace('[', '').replace(']', '').split(" "))
# face1_normed_embedding = np.array(face1['normed_embedding'].replace('[', '').replace(']', '').split(" "))

# sim = np.dot(face0_embedding, face1_embedding) / (norm(face0_embedding) * norm(face1_embedding))
# print(sim)


face0 = read_face_data("./faces_metadata/Video_1_000000_0_0_1_680_face0.json")
# face0_tmp1 = face0['embedding'].replace('[', '').replace(']', '')
# 去掉换行和空格，然后一个一个float32传给array

print(face0['embedding'])
















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
