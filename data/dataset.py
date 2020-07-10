#!/usr/bin/env python
# @Time    : 2020/7/8 10:17
# @Author  : wb
# @File    : dataset.py

'''
数据载入
'''

import torch.utils.data as data
import h5py

class CWRUDataset(data.Dataset):

    def __init__(self, filename, train):
        '''
        pytorch读取训练数据
        :param filename: 数据集文件，这边是h5py文件
        :param train: 是否为训练，还是测试
        '''
        f = h5py.File(filename, 'r')
        if train:
            self.X = f['X_train'][:]
            self.y = f['y_train'][:]
        else:
            self.X = f['X_test'][:]
            self.y = f['y_test'][:]

    def __getitem__(self, idx):
        '''
        返回一条数据
        :param idx:
        :return:
        '''
        return self.X[idx], self.y[idx]

    def __len__(self):
        '''
        数据长度
        :return:
        '''
        return self.X.shape[0]



