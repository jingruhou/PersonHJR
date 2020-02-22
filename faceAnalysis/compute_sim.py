# -*- coding: utf-8 -*-
import json
import os

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
    face_id = face_data['face_id']
    face_tmp1 = face_data['embedding'].strip('[]')
    face_tmp2 = face_tmp1.split()
    return face_id, face_tmp2


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


def compute_sim_batch(face_meta_data_file_path, faces_meta_data_dir):
    """

    :param face_meta_data_file_path: 单个face元数据
    :param faces_meta_data_dir: 需要比对的faces元数据文件夹
    :return:
    """
    sim_dict = {}

    face_meta_data = read_face_data(face_meta_data_file_path)
    face1_id, embedding1 = split_face_data(face_meta_data)
    face_embedding1_float = []  # 目标face的embedding
    for index, value in enumerate(embedding1):  # 每一个value转化为float
        item_float = float(value)
        face_embedding1_float.append(item_float)
    # print(face1_id, face_embedding1_float)

    root_dir = faces_meta_data_dir
    frame_list = os.listdir(root_dir)
    for idx in range(0, len(frame_list)):
        path = os.path.join(root_dir, frame_list[idx])
        if os.path.isfile(path):
            tmp = read_face_data(path)
            face_idx, embedding_idx = split_face_data(tmp)
            face_embedding_idx_float = []
            for index, value in enumerate(embedding_idx):
                item_float = float(value)
                face_embedding_idx_float.append(item_float)
            # print(face_idx, face_embedding_idx_float)

            # 计算相似度
            from numpy.linalg import norm
            sim_face = np.dot(face_embedding1_float, face_embedding_idx_float) / (norm(face_embedding1_float) * norm(face_embedding_idx_float))
            sim_dict[face_idx] = sim_face

    return sim_dict  # 返回目标和所有人脸特征的相似度字典列表


sim = compute_sim_batch("./faces_metadata/Video_c1_000000_0_0_1_750_face0.json", "./faces_metadata/")
print(sim)

# 按照相似度值大小进行排序 reverse=True 降序排列
sorted_sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
print(sorted_sim)
# 逐个打印key:value
# for idx in range(len(sorted_sim)):
#     print(sorted_sim[idx])

# 通过设置置信度的阈值参数
threshold = 0.5
# 比较排序后的相似度值与阈值的大小，
# 若大于阈值，则认为是同一个face；若小于阈值，则认为是不同的face
face_count = 0
for index in range(len(sorted_sim)):
    if sorted_sim[index][1] > threshold:
        print(sorted_sim[index])
        # 统计同一个face的次数
        face_count += 1
print("Face0出现的次数为： %s" % face_count)


# ########################################################################################################################
# face0 = read_face_data("./faces_metadata/Video_1_000000_0_0_1_680_face0.json")
# face0_embedding = split_face_data(face0)
# # face0_embedding格式为list，其中每一个value为str，需要转化为float32
# face0_embedding_float = []
# for index, value in enumerate(face0_embedding):
#     item_float = float(value)
#     face0_embedding_float.append(item_float)
# print(face0_embedding_float)
#
# ########################################################################################################################
# face1 = read_face_data("./faces_metadata/Video_1_000025_0_0_2_680_face0.json")
# face1_embedding = split_face_data(face1)
# # face1_embedding格式为list，其中每一个value为str，需要转化为float32
# face1_embedding_float = []
# for index, value in enumerate(face1_embedding):
#     item_float = float(value)
#     face1_embedding_float.append(item_float)
# print(face1_embedding_float)
#
# # 计算余弦相似度
# sim_embedding = compute_sim(face0_embedding_float, face1_embedding_float)
# print("相似度为：%s" % sim_embedding)
