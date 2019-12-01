"""
    抽取出所有文件名
    划分为训练集和测试集
"""

import os
import random

XIBase = '/home/lida/python_work/py_project/test/'

core_list = []
coreless_list = []

# 读取
for f in os.listdir(os.path.join(XIBase, 'core_500', 'Image')):
    core_list.append(f.split('.')[0])

for f in os.listdir(os.path.join(XIBase, 'coreless_5000', 'Image')):
    coreless_list.append(f.split('.')[0])

# 打乱
random.shuffle(core_list)
random.shuffle(coreless_list)

# 分割
train_ratio = 0.8  # 20% 测试集
core_train, core_test = core_list[:int(len(core_list)*train_ratio)],\
    core_list[int(len(core_list)*train_ratio):]
coreless_train, coreless_test = coreless_list[:int(len(coreless_list)*train_ratio)],\
    coreless_list[int(len(coreless_list)*train_ratio):]

# 保存文件
with open('sub_train_core_coreless.txt', 'w') as f:
    data = core_train
    data.extend(coreless_train)
    random.shuffle(data)
    for d in data:
        f.write('{}\n'.format(d))
with open('sub_test_core_coreless.txt', 'w') as f:
    data = core_test
    data.extend(coreless_test)
    random.shuffle(data)
    for d in data:
        f.write('{}\n'.format(d))
print('写入完成')
