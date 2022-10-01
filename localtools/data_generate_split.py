import os
import sys
import torch
import time
import pickle
import random
import collections
import math
import javalang

from torch import nn
from torch.utils import data


def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


dataset_pre_pkl_path = '../data/dataset_pre.pkl'
dataset_pre_pkl = read_pkl(dataset_pre_pkl_path)

faultType = dataset_pre_pkl.keys()

for type1 in faultType:
    # # 1.路径
    # pos_source_path = '../data/' + type1 + '/src_token_positive.txt'
    # pos_target_path = '../data/' + type1 + '/src_token_positive_patch.txt'
    # neg_source_path = '../data/' + type1 + '/src_token_negative.txt'
    # neg_target_path = '../data/' + type1 + '/src_token_negative_patch.txt'

    # # 2.读取txt文件转成list
    # pos_source = []
    # pos_target = []
    # neg_source = []
    # neg_target = []

    # with open(pos_source_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.split()
    #         pos_source.append(line)
    #     f.close()

    # with open(pos_target_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.split()
    #         pos_target.append(line)
    #     f.close()

    # with open(neg_source_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.split()
    #         neg_source.append(line)
    #     f.close()

    # with open(neg_target_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.split()
    #         neg_target.append(line)
    #     f.close()

    # print(len(pos_source), len(pos_target), len(neg_source), len(neg_target))
    source = dataset_pre_pkl[type1]['positive'] + dataset_pre_pkl[type1]['negative']
    target = dataset_pre_pkl[type1]['positive_patch'] + dataset_pre_pkl[type1]['negative_patch']

    label = []
    for i in range(len(dataset_pre_pkl[type1]['positive'])):  # 地址自动变成 0 -> len(pos_source)-1
        label.append(1)  # 定义  label[0:len(pos_source)-1]=1  指有错误的数据

    for j in range(len(dataset_pre_pkl[type1]['negative'])):
        label.append(0) # 定义  label[len(neg_source):len(neg_source)+len(pos_source)-1]=1 指没有错误的数据

    # 1.按照地址划分数据
    index = [i for i in range(len(source))]
    # random.shuffle(index)
    # print(type(index), index)

    train_scales = 0.8
    val_scales = 0.1
    test_scales = 0.1

    train_stop_flag = int(len(source) * train_scales)
    val_stop_flag = int(len(source) * (train_scales + val_scales))

    # 2.按照地址 存数据到pkl中
    train_data = {'source': [], 'target': [], 'label': []}
    valid_data = {'source': [], 'target': [], 'label': []}
    test_data = {'source': [], 'target': [], 'label': []}

    for i, j in enumerate(index):
        if (i <= train_stop_flag):
            train_data['source'].append(source[j])
            train_data['target'].append(target[j])
            train_data['label'].append(label[j])
                
        if (i > train_stop_flag and i <= val_stop_flag):
            valid_data['source'].append(source[j])
            valid_data['target'].append(target[j])
            valid_data['label'].append(label[j])
        if (i > val_stop_flag):
            test_data['source'].append(source[j])
            test_data['target'].append(target[j])
            test_data['label'].append(label[j])
    # 4.保存文件
    path = '../data/' + type1
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)
    write_pkl(train_data, path + '/train.pkl')
    write_pkl(valid_data, path + '/valid.pkl')
    write_pkl(test_data, path + '/test.pkl')

    print(type1, "数据集生成好了")
    print("train的长度为：", len(train_data['source']))
    print("valid的长度为：", len(valid_data['source']))
    print("test的长度为：", len(test_data['source']))
