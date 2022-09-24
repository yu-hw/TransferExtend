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
    f = open(filepath,"wb")
    pickle.dump(obj,f)
    f.close()

dataset_pre_pkl_path = '../data/dataset_pre.pkl'
dataset_pre_pkl = read_pkl(dataset_pre_pkl_path)

faultType = dataset_pre_pkl.keys()

for type1 in faultType:
    # 1.路径
    pos_source_path = '../data/'+type1 +'/src_token_positive.txt'
    pos_target_path = '../data/'+type1 +'/src_token_positive_patch.txt'
    neg_source_path = '../data/'+type1 +'/src_token_negative.txt'
    neg_target_path = '../data/'+type1 +'/src_token_negative_patch.txt'

    # 2.读取txt文件转成list
    pos_source = []
    pos_target = []
    neg_source = []
    neg_target = []

    with open(pos_source_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')  #删除换行符
            pos_source.append(line)
        f.close()

    with open(pos_target_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            pos_target.append(line)
        f.close()

    with open(neg_source_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            neg_source.append(line)
        f.close()

    with open(neg_target_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            neg_target.append(line)
        f.close()

    print(len(pos_source),len(pos_target),len(neg_source),len(neg_target))

    source = pos_source + neg_source
    target = pos_target + neg_target

    label = []
    for i in range(len(pos_source)):  # 地址自动变成 0 -> len(pos_source)-1
        label.append(0)  # 定义  label[0:len(pos_source)-1]=0  指有错误的数据
    # print("制作pos标签后lable的长度：",len(label))

    for j in range(len(neg_source)):
        label.append(1)  # 定义  label[len(neg_source):len(neg_source)+len(pos_source)-1]=1 指没有错误的数据

    print(len(source),len(target),len(label))


    # 1.生成3个文件夹
    path = '../data/'+type1
    take_names = ['train/', 'val/', 'test/']
    for take_name in take_names:  # 创建三个文件夹
        take_path = os.path.join(path, take_name)
        print(take_path)
        if os.path.isdir(take_path):
            pass
        else:
            os.mkdir(take_path)
    # 2.按照地址划分数据
    index = [i for i in range(len(source))]
    random.shuffle(index)
    # print(type(index), index)

    train_scales = 0.8
    val_scales = 0.1
    test_scales = 0.1

    train_stop_flage = int(len(source) * train_scales)
    val_stop_flage = int(len(source) * (train_scales + val_scales))
    # 3.按照地址 存数据到pkl中
    trian_src_list = []
    train_tgt_list = []
    trian_lab_list = []

    val_src_list = []
    val_tgt_list = []
    val_lab_list = []

    test_src_list = []
    test_tgt_list = []
    test_lab_list = []

    for i, j in enumerate(index):
        if (i <= train_stop_flage):
            trian_src_list.append(source[j])
            train_tgt_list.append(target[j])
            trian_lab_list.append(label[j])
        if (i > train_stop_flage and i <= val_stop_flage):
            val_src_list.append(source[j])
            val_tgt_list.append(target[j])
            val_lab_list.append(label[j])
        if (i > val_stop_flage):
            test_src_list.append(source[j])
            test_tgt_list.append(target[j])
            test_lab_list.append(label[j])
    # 4.保存文件
    path = '../data/' + type1
    write_pkl(trian_src_list, path + '/train/source.pkl')
    write_pkl(train_tgt_list, path + '/train/target.pkl')
    write_pkl(trian_lab_list, path + '/train/label.pkl')

    write_pkl(val_src_list, path + '/val/source.pkl')
    write_pkl(val_tgt_list, path + '/val/target.pkl')
    write_pkl(val_lab_list, path + '/val/label.pkl')

    write_pkl(test_src_list, path + '/test/source.pkl')
    write_pkl(test_tgt_list, path + '/test/target.pkl')
    write_pkl(test_lab_list, path + '/test/label.pkl')

    print("数据集生成好了")
    print("train/source的长度为：", len(read_pkl( path + '/train/source.pkl')))
    print("train/target的长度为：", len(read_pkl( path + '/train/target.pkl')))
    print("train/label的长度为：", (read_pkl( path + '/train/label.pkl')))

    print("val/source的长度为：", len(read_pkl( path + '/val/source.pkl')))
    print("val/target的长度为：", len(read_pkl( path + '/val/target.pkl')))
    print("val/label的长度为：", len(read_pkl( path + '/val/label.pkl')))

    print("test/source的长度为：", len(read_pkl( path + '/test/source.pkl')))
    print("test/target的长度为：", len(read_pkl( path + '/test/target.pkl')))
    print("test/label的长度为：", len(read_pkl( path + '/test/label.pkl')))

    print("\n\n")
