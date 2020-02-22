# coding: utf-8

import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import struct
import os

"""
    @Time    : 2020/2/10/0018 10:02
    @Author  : houjingru@semptian.com
    @FileName: insightfaceDemo.py
    @Software: PyCharm
"""


def save_feature(feature_file_name, feature_data, dim_num):
    """
    512维人脸特征embedding写入指定文件
    :param feature_file_name:
    :param feature_data:
    :param dim_num:
    :return:
    """
    with open(feature_file_name, 'wb') as fd:
        fd.write(struct.pack('i', 1))
        fd.write(struct.pack('i', dim_num))
        for i in range(len(feature_data)):
            fd.write(struct.pack('f', feature_data[i]))
        fd.close()


def save_face_metadata(metadata_file_name, metadata):
    """
    保存人脸属性信息/人物元数据
    :param metadata_file_name:
    :param metadata:
    :return: face_metadata_file.txt
        face = Face(bbox = bbox, landmark = landmark, det_score = det_score, embedding = embedding,
        gender = gender, age = age, normed_embedding=normed_embedding, embedding_norm = embedding_norm)
    """
    with open(metadata_file_name, 'w') as meta:
        meta.write(metadata)


def url_to_image(url):
    """
    网络URL请求 字节流解码为image
    :param url:
    :return:
    """
    resp = urllib.request.urlopen(url)  # 打开url
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# 6人
# url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
# 41人
# url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=114152109,1519523342&fm=26&gp=0.jpg'

# img = url_to_image(url)
"""
    sorted_sim 读取图片
"""
img = cv2.imread('C:/Users/user/PycharmProjects/PersonHJR/Resource/sorted_sim.jpg')
"""
    2 加载相关预训练模型
"""
model = insightface.app.FaceAnalysis()
ctx_id = -1  # CPU模式-初始化时通过ctx参数指定设备
# ctx_id = 0  # GPU模式-初始化时通过ctx参数指定设备
"""
    3 预设模型参数
"""
model.prepare(ctx_id=ctx_id, nms=0.4)

# x = 0
# for root, dirs, frames in os.walk("C:/Users/user/PycharmProjects/PersonHJR/Resource/Video_frames"):
#     for d in dirs:
#         print(d)  # 打印子文件夹的个数
#     for frame in frames:
#         # 读入图像
#         img_path = root + "/" + frame
#         img = cv2.imread(img_path, sorted_sim)
#
#         faces = model.get(img)
#
#         for idx, face in enumerate(faces):
#             print("人脸 [%d]:" % idx)  # 人脸编号
#             print("\t 年龄:%d" % face.age)  # 年龄
#             gender = '男'
#             if face.gender == 0:
#                 gender = '女'
#             print("\t 性别:%s" % gender)  # 性别
#             print("\t 人脸概率值:%s" % face.det_score)  # 检测概率值
#             print("\t 人脸特征维度:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
#             print("\t 人脸特征信息:%s" % face.embedding_norm)  # 嵌入人脸特征信息
#             print("\t 人脸特征信息_正则化:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
#             print("\t 人脸框:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
#             print("\t 人脸关键点:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
#             print("##############################################################################")
#             # 保存每一个人脸元数据为单个文件
#             save_face_metadata("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/"
#                                + frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(face))

"""
    4 加载图片到模型
"""
faces = model.get(img)

"""
    5 循环打印每一张人脸的相关信息 - 中文输出
"""
for idx, face in enumerate(faces):
    print("人脸 [%d]:" % idx)  # 人脸编号
    print("\t 年龄:%d" % face.age)  # 年龄
    gender = '男'
    if face.gender == 0:
        gender = '女'
    print("\t 性别:%s" % gender)  # 性别
    print("\t 人脸概率值:%s" % face.det_score)  # 检测概率值
    print("\t 人脸特征维度:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
    print("\t 人脸特征信息:%s" % face.embedding_norm)  # 嵌入人脸特征信息
    print("\t 人脸特征信息_正则化:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
    print("\t 人脸框:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
    print("\t 人脸关键点:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
    print("##############################################################################")
    # 保存每一个人脸元数据为单个文件
    save_face_metadata("sorted_sim/" + "face" + str(idx) + ".txt", str(face))





