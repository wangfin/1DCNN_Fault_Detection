#!/usr/bin/env python
# @Time    : 2021/3/15 9:58
# @Author  : wb
# @File    : t-SNE.py

'''
t-SNE可视化
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import h5py
from config import opt

class T_SNE(object):
    def __init__(self, filename):
        '''
        pytorch读取训练数据
        :param filename: 数据集文件，这边是h5py文件
        :param train: 是否为训练，还是测试
        '''
        f = h5py.File(filename, 'r')
        self.X = f['X_train'][:]
        self.y = f['y_train'][:]

        # (1218, 400) (1218,)

    def t_SNE(self):
        # (1797, 64),(1797,)
        X_tsne = TSNE(n_components=2, random_state=0).fit_transform(self.X, self.y)

        ckpt_dir = "images"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # Label_Com = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
        # ax.set_title('t-SNE')
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        # for label in tqdm(self.y):
        #     # print(label)
        #     if label == 0:
        #         s1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='plasma', label=self.y)
        #     elif label == 1:
        #         s2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='plasma', label=self.y)
        #     elif label == 2:
        #         s3 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='plasma', label=self.y)
        #     elif label == 3:
        #         s4 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='plasma', label=self.y)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='plasma', label=self.y)

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels=Label_Com, loc='upper right')

        # plt.colorbar()

        # ax.legend((s1, s2, s3, s4), ('0', '1', '2', '3'), loc='best')

        plt.savefig('images/DE_feature_0_10.png', dpi=400)
        plt.show()

if __name__ == '__main__':
    tsne = T_SNE('DE_feature_0_10.h5')
    tsne.t_SNE()
