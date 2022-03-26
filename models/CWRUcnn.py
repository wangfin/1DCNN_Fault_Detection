#!/usr/bin/env python
# @Time    : 2020/7/8 15:47
# @Author  : wb
# @File    : CWRUcnn.py

'''
一维CNN的模型
'''
import torchsnooper
from torch import nn
from .BasicModule import BasicModule

class Flatten(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CWRUcnn(BasicModule):

    def __init__(self, kernel1=27, kernel2=36, kernel_size=10, pad=0, ms1=4, ms2=4):
        super(CWRUcnn, self).__init__()
        self.model_name = 'CWRUcnn'

        self.conv = nn.Sequential(
            nn.Conv1d(1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel1, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel2, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 10),
            nn.ReLU(),
            # nn.Linear(9, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        self.feature = x
        x = self.fc(x)
        return x
















