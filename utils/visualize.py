#!/usr/bin/env python
# @Time    : 2020/7/9 20:44
# @Author  : wb
# @File    : visualize.py

'''
绘图程序
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

class Visualizer(object):

    def draw_data(self, matfilepath, length):
        '''
        把振动数据绘制成波形图
        :param matfilepath:mat文件路径
        :param length:绘制的长度
        :return:
        '''

        # 97（正常数据）209（内）222（球）234（外）
        # Normal;Inner raceway Fault;Roll Fault;Outer raceway Fault
        # 读取raw_data中的mat文件图像
        raw_data = scio.loadmat(matfilepath)
        # 读取内容
        signal = ''
        for key, value in raw_data.items():
            if key[5:7] == 'DE':
                signal = value
        # print(type(signal))

        time = [i for i in range(length)]

        font = {'family': 'Times New Roman', 'size': 18}

        plt.figure(figsize=(10, 5))  # 设置画布的尺寸
        plt.plot(time, signal[:length], color='mediumblue')
        plt.xlabel("")
        plt.ylabel("", font)
        plt.yticks(fontproperties='Times New Roman', size=18)
        plt.xticks(fontproperties='Times New Roman', size=18)
        # plt.title("Normal")
        plt.savefig('../data/Normal.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    view = Visualizer()

    matfilepath = '../data/raw_data/97.mat'
    view.draw_data(matfilepath, 400)




















