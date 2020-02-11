# coding: utf-8

import insightface
import urllib
import urllib.request
import cv2
import numpy as np

"""
    @Time    : 2020/2/10/0018 10:02
    @Author  : houjingru@semptian.com
    @FileName: insightfaceDemo.py
    @Software: PyCharm
"""


def url_to_image(url):
    """
    网络URL请求 字节流解码为image
    :param url:
    :return:
    """
    resp = urllib.request.urlopen(url)  # 打开url
    image = np.asarray(bytearray(resp.read()), dtype="uint8")  #
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  #
    return image


# 6人
# url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
# 41人
# url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=114152109,1519523342&fm=26&gp=0.jpg'

# img = url_to_image(url)
"""
    1 读取图片
"""
img = cv2.imread('C:/Users/user/PycharmProjects/PersonHJR/Resource/HJR.jpg')
"""
    2 加载相关预训练模型
"""
model = insightface.app.FaceAnalysis()
ctx_id = -1
"""
    3 预设模型参数
"""
model.prepare(ctx_id=ctx_id, nms=0.4)
"""
    4 加载图片到模型
"""
faces = model.get(img)
"""
    5 循环打印每一张人脸的相关信息
"""
for idx, face in enumerate(faces):
    print("Face [%d]:" % idx)  # 人脸编号
    print("\t age:%d" % (face.age))  # 年龄
    gender = 'Male'
    if face.gender == 0:
        gender = 'Female'
    print("\t gender:%s" % (gender))  # 性别
    print("\t det_score:%s" % face.det_score)  # 检测概率值
    print("\t embedding shape:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
    print("\t embedding embedding_norm:%s" % face.embedding_norm)  # 嵌入人脸特征信息（）
    print("\t embedding normed_embedding:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
    print("\t bbox:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
    print("\t landmark:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
    print("##############################################################################")

"""
    5 循环打印每一张人脸的相关信息 - 中文输出
"""
for idx, face in enumerate(faces):
    print("人脸 [%d]:" % idx)  # 人脸编号
    print("\t 年龄:%d" % (face.age))  # 年龄
    gender = '男'
    if face.gender == 0:
        gender = '女'
    print("\t 性别:%s" % (gender))  # 性别
    print("\t 人脸概率值:%s" % face.det_score)  # 检测概率值
    print("\t 人脸特征维度:%s" % face.embedding.shape)  # 嵌入人脸特征的维度
    print("\t 人脸特征信息:%s" % face.embedding_norm)  # 嵌入人脸特征信息（）
    print("\t 人脸特征信息_正则化:\n%s" % face.normed_embedding)  # 嵌入人脸特征信息（正则化）
    print("\t 人脸框:%s" % (face.bbox.astype(np.int).flatten()))  # 人脸框大小
    print("\t 人脸关键点:%s" % (face.landmark.astype(np.int).flatten()))  # 人脸关键点坐标值
    print("##############################################################################")