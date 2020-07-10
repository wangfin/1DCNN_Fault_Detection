#!/usr/bin/env python
# @Time    : 2020/7/8 15:37
# @Author  : wb
# @File    : BasicModule.py

'''
封装了nn.Module，主要提供save和load两个方法
'''

import torch
import time

class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(self)  # 默认名字
        # print(self.model_name)

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        else:
            prefix = 'checkpoints/' + name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name












