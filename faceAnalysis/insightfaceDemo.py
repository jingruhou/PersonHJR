# coding: utf-8

import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import struct
import os
import json
import pickle

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
    :param metadata_file_name:写入的文件名
    :param metadata:写入的数据
    :return: face_metadata_file.txt （纯文本，无结构，难以解析获取单个值）
        face = Face(bbox = bbox, landmark = landmark, det_score = det_score, embedding = embedding,
        gender = gender, age = age, normed_embedding=normed_embedding, embedding_norm = embedding_norm)
    """
    with open(metadata_file_name, 'w') as meta:
        meta.write(metadata)


def save_face_meta_data_standard(metadata_file_name, metadata):
    """
    保存人脸属性信息/人物元数据 - 标准化
    :param metadata_file_name:写入的文件名
    :param metadata:写入的数据
    :return: face_meta_data_file.txt （list，每一个元属性为一行list[index]）
        bbox = bbox,
        landmark = landmark,
        det_score = det_score,
        embedding = embedding,
        gender = gender,
        age = age,
        normed_embedding=normed_embedding,
        embedding_norm = embedding_norm
    Author:2020年2月18日 houjingru@semptian.com
    """
    with open(metadata_file_name, "a") as meta:
        meta.write(metadata)
        meta.write("\n")


def save_face_meta_data_json(metadata_file_name, dict_face_meta_data):
    """
    保存人脸属性信息/人物元数据 - JSON文件
    :param metadata_file_name: 写入的文件名
    :param dict_face_meta_data: 写入的数据-字典格式
    :return: face_meta_data_file.json

    Author:2020年2月19日 houjingru@semptian.com
    """
    with open(metadata_file_name, "w", encoding="utf-8") as file:
        file.write(dict_face_meta_data)


def url_to_image(url):
    """
    网络URL请求 字节流解码为image
    :param url:
    :return:
    """
    resp = urllib.request.urlopen(url)  # 打开url
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# 6人
# url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
# 41人
# url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=114152109,1519523342&fm=26&gp=0.jpg'

# img = url_to_image(url)

"""
    1 读取图片
"""
# img = cv2.imread('C:/Users/user/PycharmProjects/PersonHJR/Resource/XJP6.jpg')
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
"""
    4 加载图片文件夹到模型,并且循环打印结果
"""
x = 0
# os.walk()返回结果：文件夹根路径dir_path, 文件夹名称dir_names, 文件名称file_names
for root, dirs, frames in os.walk("C:/Users/user/PycharmProjects/PersonHJR/Resource/Video_frames"):
    for d in dirs:
        print(d)  # 打印子文件夹的个数
    for frame in frames:
        # 读入图像
        img_path = root + "/" + frame
        img = cv2.imread(img_path, 1)

        faces = model.get(img)

        for idx, face in enumerate(faces):
            # print("人脸 [%d]:" % idx)  # 人脸编号
            # print("\t 年龄:%d" % face.age)  # 年龄
            # gender = '男'
            # if face.gender == 0:
            #     gender = '女'
            # print("\t 性别:%s" % gender)  # 性别
            # print("\t 人脸概率值:%s" % face.det_score)  # 检测概率值
            # print("\t 人脸特征:%s" % face.embedding)  # 嵌入人脸特征信息
            # print("\t 人脸特征维度:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
            # print("\t 人脸特征信息:%s" % face.embedding_norm)  # 嵌入人脸特征L2范式
            # print("\t 人脸特征信息_正则化:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
            # print("\t 人脸框:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
            # print("\t 人脸关键点:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
            # print("##############################################################################")

            # 构造人俩属性字典
            dict_face_meta = {'face_id': frame.split('.')[0] + "_" + "face" + str(idx),
                              'age': str(face.age),
                              'gender': str(face.gender),
                              'bbox': str(face.bbox.astype(np.float).flatten()),
                              'landmark': str(face.landmark.astype(np.float).flatten()),
                              'det_score': str(face.det_score),
                              'embedding': str(face.embedding),
                              'embedding_shape': str(face.embedding.shape),
                              'embedding_norm': str(face.embedding_norm),
                              'normed_embedding': str(face.normed_embedding)
                              }
            # face_meta_data = json.dumps(dict_face_meta)
            # indent=8 缩进，ensure_ascii=False不使用ascii编码，即可以显示中文内容
            face_meta_data = json.dumps(dict_face_meta, indent=8, ensure_ascii=False)
            print(face_meta_data)
            with open("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" + frame.split('.')[0] + "_" + "face" + str(idx) + ".json", "w", encoding="utf-8") as f:
                f.write(face_meta_data)

            # save_face_meta_data_json("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" + frame.split('.')[0] + "_" + "face" + str(idx) + ".json", dict_face_meta)

            # 保存每一个人脸元数据为单个文件
            # 优化部分：将人脸的属性数据转化为str，计算相似度的时候会不会有影响？
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              "face_id:" + frame.split('.')[0] + "_" + "face" + str(idx) + "\n" +
            #                              "age:" + str(face.age) + "\n" +
            #                              "gender:" + str(face.gender) + "\n" +
            #                              "bbox:" + str(face.bbox.astype(np.float).flatten()) + "\n" +
            #                              "landmark:" + "\n" + str(face.landmark.astype(np.float).flatten()) + "\n" +
            #                              "det_score:" + str(face.det_score) + "\n" +
            #                              "embedding:" + "\n" + str(face.embedding) + "\n" +
            #                              "embedding_shape:" + str(face.embedding.shape) + "\n" +
            #                              "embedding_norm:" + str(face.embedding_norm) + "\n" +
            #                              "normed_embedding:" + "\n" + str(face.normed_embedding))

            # save_face_meta_data_json("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" + frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", face)

            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(idx))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(face.age))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(face.gender))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(face.det_score))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              str(face.embedding.shape))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              str(face.embedding_norm))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              str(face.normed_embedding))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              str(face.bbox.astype(np.int).flatten()))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" +
            #                              frame.split('.')[0] + "_" + "face" + str(idx) + ".txt",
            #                              str(face.landmark.astype(np.int).flatten()))

            # save_face_metadata("sorted_sim/" + "face" + str(idx) + ".txt", str(face))
            # save_face_meta_data_standard("C:/Users/user/PycharmProjects/PersonHJR/faceAnalysis/faces_metadata/" + frame.split('.')[0] + "_" + "face" + str(idx) + ".txt", str(face))


# """
#     5 循环打印每一张人脸的相关信息 - 英文输出
# """
# for idx, face in enumerate(faces):
#     print("Face [%d]:" % idx)  # 人脸编号
#     print("\t age:%d" % face.age)  # 年龄
#     gender = 'Male'
#     if face.gender == 0:
#         gender = 'Female'
#     print("\t gender:%s" % gender)  # 性别
#     print("\t det_score:%s" % face.det_score)  # 检测概率值
#     print("\t embedding shape:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
#     print("\t embedding embedding_norm:%s" % face.embedding_norm)  # 嵌入人脸特征信息（）
#     print("\t embedding normed_embedding:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
#     print("\t bbox:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
#     print("\t landmark:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
#     print("##############################################################################")

# """
#    5 循环打印每一张人脸的相关信息 - 中文输出
# """
# for idx, face in enumerate(faces):
#     print("人脸 [%d]:" % idx)  # 人脸编号
#     print("\t 年龄:%d" % face.age)  # 年龄
#     gender = '男'
#     if face.gender == 0:
#         gender = '女'
#     print("\t 性别:%s" % gender)  # 性别
#     print("\t 人脸概率值:%s" % face.det_score)  # 检测概率值
#     print("\t 人脸特征维度:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
#     print("\t 人脸特征信息:%s" % face.embedding_norm)  # 嵌入人脸特征信息
#     print("\t 人脸特征信息_正则化:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
#     print("\t 人脸框:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
#     print("\t 人脸关键点:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
#     print("##############################################################################")
#     # 保存每一个人脸元数据为单个文件
#     save_face_metadata("XJP6/" + "face" + str(idx) + ".txt", str(face))


def compute_sim_norm(img1_norm, img2_norm):
    """
    通过计算两张图片的L2范式值，求其相似度
    :param img1_norm:
    :param img2_norm:
    :return:
    """


def compute_sim(img1, img2):
    """
    余弦相似度计算（Cosine Similarity） - 夹角余弦
    :param img1: 图片1
    :param img2: 图片2
    :return: 相似度

    Note1:余弦相似度取值[-sorted_sim， sorted_sim]，值趋于1，表示两个向量的相似度越高。
        余弦相似度与向量的幅值无关，只与向量的方向相关

    Note2:余弦相似度用向量空间中两个向量夹角的余弦值作为衡量两个个体间差异的大小。
        相比距离度量，余弦相似度更加注重两个向量在方向上的差异，而非距离或者长度上。
    """
    emb1 = model.rec_model.get_embedding(img1).flatten()
    emb2 = model.rec_model.get_embedding(img2).flatten()
    from numpy.linalg import norm

    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

    return sim
