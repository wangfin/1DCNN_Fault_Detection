#!/usr/bin/env python
# @Time    : 2020/7/8 14:41
# @Author  : wb
# @File    : config.py

'''
配置文件
包括文件配置
模型参数等
'''
import os
import warnings

class Config(object):
    env = 'default'  # visdom 环境
    model = 'CWRUcnn'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root = 'data'
    mat_root = 'raw_data'  # mat数据文件根目录
    list_filename = 'annotations_4.txt'  # mat文件的文件列表
    h5filename = 'DE_3_4.h5'
    feature_filename = 'data/DE_feature_0_10.h5'

    train_data_root = 'data/DE_0_10.h5'
    val_data_root = 'data/DE_0_10.h5'
    test_data_root = 'data/DE_0_10.h5'

    dim = 400  # 数据的维度
    train_fraction = 0.8  # 训练集所占的占比
    split_num = 2
    category = 10  # 类别数量

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    print_freq = 20  # print info every N batch
    device = 'cuda:0'
    print_every = 100

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = './results/confuse_matrix_rate.xlsx'
    load_model_path = './checkpoints/CWRUcnn_0325_21_00_40.pth'

    max_epoch = 20
    lr = 0.001  # initial learning rate
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    lr_decay_iters = 1  # 每一轮都减少lr
    weight_decay = 1e-4  # 损失函数


# def parse(self, kwargs):
#     '''
#     通过命令行的方式修改默认的参数
#     根据字典kwargs 更新 config参数
#     '''
#     for k, v in kwargs.iteritems():
#         if not hasattr(self, k):
#             warnings.warn("Warning: opt has not attribut %s" % k)
#         setattr(self, k, v)
#
#     print('user config:')
#     for k, v in self.__class__.__dict__.iteritems():
#         if not k.startswith('__'):
#             print(k, getattr(self, k))

# Config.parse = parse

opt = Config()

# opt.parse = parse













