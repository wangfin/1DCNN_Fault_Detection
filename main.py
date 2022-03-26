#!/usr/bin/env python
# @Time    : 2020/7/8 16:07
# @Author  : wb
# @File    : main.py

'''
主程序
包括训练，测试等功能
'''
import h5py

from config import opt
from torch.utils.data import DataLoader
from data.dataset import CWRUDataset
import models
import torch
from tensorboardX import SummaryWriter
import copy
import time
import pandas as pd
import numpy as np
from torchsummary import summary
import torchsnooper


def train():
    '''
    训练模块
    :param kwargs:
    :return:
    '''
    # opt.parse(kwargs)

    # step1: 模型
    model = getattr(models, opt.model)()
    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    # summary(model, (1, 2048))

    # step2: 数据
    train_data = CWRUDataset(opt.train_data_root, train=True)
    val_data = CWRUDataset(opt.val_data_root, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters,
                                                opt.lr_decay)  # regulation rate decay

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    # loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)
    # previous_loss = 1e100

    writer = SummaryWriter()
    # 保存准确率最高的模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = ''
    if opt.use_gpu:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA is available')
            device = torch.device(opt.device)
        else:
            device = torch.device('cpu')

    # train
    for epoch in range(opt.max_epoch):

        start_time = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, opt.max_epoch))
        # set train model or val model for BN and Dropout layers
        model.train()

        for ii, (data, label) in enumerate(train_dataloader):

            # add one dim to fit the requires of conv1d layer
            # print('x size', data.size())
            data.resize_(data.size()[0], 1, data.size()[1])
            # data.resize_(data.size()[0], data.size()[1], 1)
            data, label = data.float(), label.long()
            input, target = data.to(device), label.to(device)

            optimizer.zero_grad()
            score = model(input)
            # 求loss
            loss = criterion(score, target)
            writer.add_scalar('loss', loss.item())
            # print and save loss per 'print_every' times
            if (ii + 1) % opt.print_every == 0:
                print('t = %d, loss = %.4f' % (ii + 1, loss.item()))

            loss.backward()
            # 优化参数
            optimizer.step()
            # 修改学习率
            scheduler.step()

        # save epoch loss and acc to train or val history
        train_acc, _ = check_accuracy(model, train_dataloader, device)
        val_acc, _ = check_accuracy(model, val_dataloader, device)
        # writer acc and weight to tensorboard
        writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        end_time = time.time()
        print('训练时间：', end_time - start_time)

    # 加载最佳的参数
    model.load_state_dict(best_model_wts)
    val_acc, confuse_matrix = check_accuracy(model, val_dataloader, device, error_analysis=True)
    # 将混淆矩阵写入Excel
    data_pd = pd.DataFrame(confuse_matrix)
    writer = pd.ExcelWriter(opt.result_file)
    data_pd.to_excel(writer)
    writer.save()
    writer.close()

    # 保存模型
    model_save_path = model.save(opt.model)
    print('最优模型保存在：', model_save_path)


def test():

    # 模型文件
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)

    # 加载数据集
    test_dataset = CWRUDataset(opt.test_data_root, train=False)
    test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False)
    print('testing length = %d' % len(test_dataset))

    device = ''
    if opt.use_gpu:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA is available')
            device = torch.device(opt.device)
        else:
            device = torch.device('cpu')

    model = model.to(device)
    # 保存标签
    f = h5py.File(opt.feature_filename, 'w')
    print('y_shape', test_dataset.y.shape)
    f.create_dataset('y_train', data=test_dataset.y)

    # 测试，并保存中间特征
    test_acc, confuse_matrix = check_accuracy(model, test_loader, device, feature_file=f, error_analysis=True)

    f.close()

def check_accuracy(model, loader, device, feature_file, error_analysis=False):
    '''
    检查模型的准确率，如果错误分析返回混淆矩阵
    :param model:模型文件
    :param loader:数据加载
    :param device:设备
    :param error_analysis:
    :return:
    '''
    X_feature = np.empty([0, 36])
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    times = []
    with torch.no_grad():
        # one batch
        for x, y in loader:
            # print('x_size', x.size())
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)

            # 记录测试时间
            start_time = time.time()
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            end_time = time.time()
            times.append(end_time - start_time)
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))

            # 保存中间特征与标签
            feature_output = model.feature.cpu()
            # X_feature = np.append(X_feature, feature_output)
            X_feature = np.concatenate((X_feature, feature_output), axis=0)

        # 保存特征
        print('X_shape', X_feature.shape)
        feature_file.create_dataset('X_train', data=X_feature)

    sum_time = 0
    for i in times:
        sum_time = sum_time + i
    # print('测试时间', sum_time)
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix


if __name__=='__main__':
    # train()
    test()

