# coding: utf-8
import mxnet as mx
"""
    @Time    : 2020/2/18/0018 14:21
    @Author  : houjingru@semptian.com
    @FileName: ndarray.py
    @Software: PyCharm
"""
x = mx.nd.ones((2, 3))
y = x.asnumpy()
print(type(y))
print(y)

z = mx.nd.ones((2, 3), dtype='int32')
print(z.asnumpy())
