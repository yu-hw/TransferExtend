import os
import torch
import pickle

from torch import nn
from torch.utils import data

import module.model as model
import module.vocab as vocab
import module.data as data
import module.optimizer as optimizer
import module.dataloader as datalodaer
import setting
import utils

# from d2l import torch as d2l

#  1、train文件中需要将每一个功能都分开在不同的模块文件中 √
#  2、需要修改数据集，总共有三种数据 两个标签 neg数据中的rank关键字删去，生成 source、target、label三个数据 √
#  3、需要设置训练集、验证集、测试集。 √
#  4、整理代码规范 和 生成11个文件中的 train\val\test √
#  5、制作 train、val、test的数据 迭代器 √
#  6、写loss函数
#  7、将数据输入到 模型中，计算损失


def load_data(opt):
    return data.load_data(opt)


def build_vocab(opt, data): 
    src = data['train']['source'] + data['valid']['source'] + data['test']['source']
    tgt = data['train']['target'] + data['valid']['target'] + data['test']['target']
    return vocab.build_vocab(opt, src, tgt)


def build_iterator(opt, data):
    return datalodaer.build_dataloader(opt, data)


def build_net(opt):
    return model.buildSeq2SeqModel(opt)


def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)

# def build_loss():
#     raise NotImplementedError
#
# def train():
#     raise NotImplementedError


def main():
    opt = setting.get_opt()
    
    device = opt['device'] = utils.get_device()
    
    data = load_data(opt)
    
    src_vocab, tgt_vocab = build_vocab(opt, data)
    
    train_iter = build_iterator(opt, data['train'])
    valid_iter = build_iterator(opt, data['valid'])
    test_iter = build_iterator(opt, data['test'])
    
    model = build_net(opt)
    print("使用的模型为：", model)
    
    parameters = model.parameters()
    print("参数个数为：", utils.count_parameters(model))

    raise NotImplementedError
    
    # 6.优化器、损失函数
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    # 7.训练
    print('Start training ...')
    model.to(model.device)
    model.train()


if __name__ == '__main__':
    main()
