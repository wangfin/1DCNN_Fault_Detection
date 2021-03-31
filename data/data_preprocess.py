#!/usr/bin/env python
# @Time    : 2020/7/8 14:34
# @Author  : wb
# @File    : data_preprocess.py

'''
数据预处理文件
处理原始的mat文件，将mat文件转换为h5py保存的数据

将CWRU原始数据分为驱动端与风扇端（DE，FE），根据转速与故障将数据分为101种类别，
其中训练样本80%，测试样本20%，随机打乱取样。
最后数据以h5格式存储，其中DE为驱动端测点的数据集，包含训练样本与测试样本；FE为风扇端。

只需要97（正常数据）209（内）222（球）234（外）这四个数据
读入数据之后将按400个采样点进行切分
'''

import os
import numpy as np
import pandas as pd
import scipy.io as scio
import h5py
from config import opt

class DataProcess():

    def __init__(self, root, list_filename):
        '''
        读取数据集的基本信息
        :param root: 数据集根目录
        :param list_filename: 因为是多个文件，所以需要一张文件列表
        '''
        self.root = root
        if list_filename != None:
            self.frame = pd.read_table(list_filename)
        else:
            raise ValueError("file_list [%s] not recognized." % list_filename)

    def process(self, dim, train_fraction):
        '''
        处理数据集，划分训练集，测试集
        :param dim: 需要的维度
        :param train_fraction: 划分训练集的比例
        :return:
        '''

        # 设置空数组
        signals_tr, signals_te, labels_tr, labels_te = [], [], [], []

        for idx in range(len(self.frame)):
            mat_name = os.path.join(self.root, self.frame['file_name'][idx])
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            # print(raw_data)
            for key, value in raw_data.items():
                if key[5:7] == 'DE':
                    signal = value

                    # dim个数据点一个划分，计算有多少个数据块
                    sample_num = signal.shape[0] // dim

                    # 划分训练集和测试集（根据数据块）
                    train_num = int(sample_num * train_fraction)
                    test_num = sample_num - train_num

                    signal = signal[0:dim * sample_num]
                    # 按sample_num切分，每个dim大小
                    signals = np.array(np.split(signal, sample_num))

                    # 数据样本和标签
                    signals_tr.append(signals[0:train_num, :])
                    signals_te.append(signals[train_num:sample_num, :])

                    # 因为需求，所以需要修改label
                    labels_tr.append(idx * np.ones(int(train_num)))
                    labels_te.append(idx * np.ones(int(test_num)))

        # 最后将数据拼接在一起
        signals_tr_np = np.concatenate(signals_tr).squeeze()  # 纵向的拼接，删除维度为1的维度
        labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
        signals_te_np = np.concatenate(signals_te).squeeze()
        labels_te_np = np.concatenate(np.array(labels_te)).astype('uint8')

        # print(signals_te_np)
        # print(labels_te_np)

        print(signals_tr_np.shape, labels_tr_np.shape, signals_te_np.shape, labels_te_np.shape)

        return signals_tr_np, labels_tr_np, signals_te_np, labels_te_np

    # def create_dataset(self, dim, train_fraction, split_num):
    #     '''
    #     将每个文件的数据按照每400个点切分，将其中每40个点拆出来（10段）
    #     然后每一段的第一个40，第二个40...全部保存起来
    #     最后应该是10个文件，每个文件里是40*1218
    #     :return:
    #     '''
    #
    #     # 设置空数组
    #     signals_tr, signals_te, labels_tr, labels_te = [], [], [], []
    #
    #     # 读取原始的数据集
    #     for idx in range(len(self.frame)):
    #         mat_name = os.path.join(self.root, self.frame['file_name'][idx])
    #         raw_data = scio.loadmat(mat_name)
    #
    #         # print(raw_data)
    #
    #         # 将原始数据集进行拆分
    #         for key, value in raw_data.items():
    #             if key[5:7] == 'DE':
    #                 signal = value
    #                 # print(signal.shape)
    #                 # 40个数据点一个划分，计算有多少个数据块
    #                 sample_num = signal.shape[0] // dim
    #
    #                 # 分成十段
    #                 for x in range(split_num):
    #                     split_data = []
    #                     # 拆分数据集
    #                     for i in range(sample_num):
    #                         if i % split_num == x:
    #                             # print(i, x)
    #                             split_data.append(signal[i*dim:(i+1)*dim])
    #                     # print(len(split_data))
    #                     # print('sp', split_data)
    #
    #                     # 把list转换为numpy数组
    #                     split_data = np.array(split_data)
    #                     # print(split_data)
    #                     # print(split_data.shape)
    #
    #                     # 划分训练集和测试集（根据数据块）
    #                     train_num = int(len(split_data) * train_fraction)
    #                     test_num = len(split_data) - train_num
    #
    #                     # 数据样本和标签
    #                     signals_tr.append(split_data[0:train_num, :])
    #                     signals_te.append(split_data[train_num:train_num+test_num, :])
    #
    #                     # 数据label
    #                     labels_tr.append((idx*split_num+x) * np.ones(int(train_num)))
    #                     labels_te.append((idx*split_num+x) * np.ones(int(test_num)))
    #
    #                     print(labels_te)
    #
    #     # 最后将数据拼接在一起
    #     signals_tr_np = np.concatenate(signals_tr).squeeze()  # 纵向的拼接，删除维度为1的维度
    #     labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
    #     signals_te_np = np.concatenate(signals_te).squeeze()
    #     labels_te_np = np.concatenate(np.array(labels_te)).astype('uint8')
    #
    #     # print(labels_te_np)
    #
    #     print(signals_tr_np.shape, labels_tr_np.shape, signals_te_np.shape, labels_te_np.shape)
    #
    #     return signals_tr_np, labels_tr_np, signals_te_np, labels_te_np

    def save(self, filename, X_train, y_train, X_test, y_test):
        '''
        将上面处理好的文件保存为h5文件
        :param filename:保存的文件名
        :param X_train:输入训练集
        :param y_train:训练集标签
        :param X_test:输入测试集
        :param y_test:测试集标签
        :return:
        '''
        f = h5py.File(filename, 'w')
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()

if __name__ == '__main__':
    Dp = DataProcess(opt.mat_root, opt.list_filename)
    X_train, y_train, X_test, y_test = Dp.process(opt.dim, opt.train_fraction)
    Dp.save(opt.h5filename, X_train, y_train, X_test, y_test)
    print('h5文件保存成功')

    # X_train, y_train, X_test, y_test = Dp.create_dataset(opt.dim, opt.train_fraction, opt.split_num)
    # Dp.save(opt.h5filename, X_train, y_train, X_test, y_test)
    # print('h5文件保存成功')



