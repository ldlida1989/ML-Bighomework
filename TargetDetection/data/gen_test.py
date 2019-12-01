"""
    抽取出所有文件名
    划分为训练集和测试集
"""

import os
import random

XIBase = '/home/lida/python_work/py_project/test/'

images = []

# 读取
for f in os.listdir(os.path.join(XIBase, 'Image')):
    images.append(f.split('.')[0])

with open('sub_test_core_coreless1.txt', 'w') as f:
    for d in images:
        f.write('{}\n'.format(d))
print('写入完成')
