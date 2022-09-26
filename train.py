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
import module.loss as loss
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
    src = data['train']['source'] + \
        data['valid']['source'] + data['test']['source']
    tgt = data['train']['target'] + \
        data['valid']['target'] + data['test']['target']
    return vocab.build_vocab(opt, src, tgt)


def text2id(data, src_vocab, tgt_vocab):
    data['train']['source'] = src_vocab[data['train']['source']]
    data['train']['target'] = tgt_vocab[data['train']['target']]
    data['valid']['source'] = src_vocab[data['valid']['source']]
    data['valid']['target'] = tgt_vocab[data['valid']['target']]
    data['test']['source'] = src_vocab[data['test']['source']]
    data['test']['target'] = tgt_vocab[data['test']['target']]

def add_bos_eos(data,opt):
    [l.append(opt['vocab']['tgt_eos']) for l in data['train']['target']]
    [l.append(opt['vocab']['tgt_eos']) for l in data['valid']['target']]
    [l.append(opt['vocab']['tgt_eos']) for l in data['test']['target']]

    [l.insert(0,opt['vocab']['tgt_bos']) for l in data['train']['target']]
    [l.insert(0,opt['vocab']['tgt_bos']) for l in data['valid']['target']]
    [l.insert(0,opt['vocab']['tgt_bos']) for l in data['test']['target']]

def truncate_pad(data, num_steps, padding_token):
    if len(data) > num_steps:
        return data[:num_steps]  # 截断
    return data + [padding_token] * (num_steps - len(data))  # 填充

def align_data(data, num_steps, padding_token):
    data['train']['source'] = [truncate_pad(l, num_steps, padding_token) for l in data['train']['source']]
    train_src_valid_len = [ (num_steps - l.count(padding_token)) for l in data['train']['source']]
    data['train']['target'] = [truncate_pad(l, num_steps, padding_token) for l in data['train']['target']]
    train_tgt_valid_len = [ (num_steps - l.count(padding_token)) for l in data['train']['target']]

    data['valid']['source'] = [truncate_pad(l, num_steps, padding_token) for l in data['valid']['source']]
    valid_src_valid_len = [ (num_steps - l.count(padding_token)) for l in data['valid']['source']]
    data['valid']['target'] = [truncate_pad(l, num_steps, padding_token) for l in data['valid']['target']]
    valid_tgt_valid_len = [(num_steps - l.count(padding_token)) for l in data['valid']['target']]

    data['test']['source'] = [truncate_pad(l, num_steps, padding_token) for l in data['test']['source']]
    test_src_valid_len = [ (num_steps - l.count(padding_token)) for l in data['test']['source']]
    data['test']['target'] = [truncate_pad(l, num_steps, padding_token) for l in data['test']['target']]
    test_tgt_valid_len = [(num_steps - l.count(padding_token)) for l in data['test']['target']]

    return train_src_valid_len,train_tgt_valid_len,valid_src_valid_len,valid_tgt_valid_len,test_src_valid_len,test_tgt_valid_len

def build_iterator(opt, data):
    return datalodaer.build_dataloader(opt, data)


def build_net(opt):
    return model.buildSeq2SeqModel(opt)


def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)


def build_loss(opt):
    return loss.build_loss_seq2seq(opt)


def train_step(opt, net, iterator, optimizer, ctiterion):
    device = opt['device']
    
    model.to(device)
    model.train()
    
    for data in iterator:
        src, tgt, label, src_len, tgt_len = data
        optimizer.zero_grad()
        outs = net(opt, src, tgt, src_len, tgt_len)
        l = ctiterion(outs, tgt[1:])
        l.sum().backward()
        utils.grad_clipping(net, 1)
        predict_num_tokens = tgt_len.sum() - len(tgt_len) # 去掉 <bos>
        optimizer.step()
        # 补充一个统计用模块


def main():
    print("### Load option")
    opt = setting.get_opt()
    opt['device'] = utils.get_device()

    print("### Load data")
    #  加载的pkl数据格式为： data['train'、'valid'、'test']['source'、'target'、‘label’]
    #      返回的数据为:['public', 'void', 'start', '(', ')', '{', 'try', '{', 'if',....]
    data = load_data(opt)
    print("其中一个原数据为：",data['train']['source'][1])

    print("### Build vocabulary")
    src_vocab, tgt_vocab = build_vocab(opt, data)

    print("### Convert text to id")
    text2id(data, src_vocab, tgt_vocab)
    print("转化为数值数据为：\n",data['train']['target'][1],'\n',data['valid']['target'][1],'\n',data['test']['target'][1])


    print("### add tgt_bos tgt_eos to target")
    add_bos_eos(data,opt)
    print("\n target中增加eos bos后转化为数值数据为：\n",data['train']['target'][1],'\n',data['valid']['target'][1],'\n',data['test']['target'][1])


    print("### truncate and pad data  and count valid len")
    train_src_valid_len,train_tgt_valid_len,valid_src_valid_len,\
    valid_tgt_valid_len,test_src_valid_len,test_tgt_valid_len \
        = align_data(data, opt['num_steps'] , src_vocab['<pad>'])
    print("\n填充pad后的src数据为：\n", data['train']['source'][6], "\n填充pad后的tgt数据为：\n", data['train']['target'][6])
    # print("有效长度：",train_src_valid_len,train_tgt_valid_len,valid_src_valid_len,\
    # valid_tgt_valid_len,test_src_valid_len,test_tgt_valid_len)

    print("### generate data['train', 'valid', 'test']['source', 'target', 'label', 'source_length', 'target_length']  ")
    data['train']['source_length'] = train_src_valid_len
    data['train']['target_length'] = train_tgt_valid_len
    data['valid']['source_length'] = valid_src_valid_len
    data['valid']['target_length'] = valid_tgt_valid_len
    data['test']['source_length'] = test_src_valid_len
    data['test']['target_length'] = test_tgt_valid_len

    print("data数据的key是：", data.keys())
    print("data['train']数据的key是：",data['train'].keys())

    # 此时 data 是数字形式
    # 首先需要在 target 前后加上 <bos>, <eos>
    # 即 opt['vocab']['tgt_bos'], opt['vocab']['tgt_eos']
    # 然后需要将数字形式填充截断到固定长度
    # source 使用 opt['vocab']['src_pad']
    # target 使用 opt['vocab']['tgt_pad']
    # 长度设定为 opt['padding_length']
    # 传回的数据形式为：
    ## data['train', 'valid', 'test']['source', 'target', 'label', 'source_length', 'target_length']

    print("### Build iterator")
    train_iter = build_iterator(opt, data['train'])
    valid_iter = build_iterator(opt, data['valid'])
    test_iter = build_iterator(opt, data['test'])
    for src, tgt, label, src_len, tgt_len in train_iter:
        print("scr:\n",src)
        print("tgt:\n",tgt)
        print("label:\n",label)
        print("src valid len:\n",src_len)
        print("tgt valid len:\n",tgt_len)
        break


    print("### Build net")
    model = build_net(opt)
    print("使用的模型为：", model)

    parameters = model.parameters()
    print("参数个数为：", utils.count_parameters(model))

    # 6.优化器、损失函数
    print("### Build optimizer and loss")
    optimizer = build_optimizer(opt, model)
    ctiterion = build_loss(opt)

    # 7.训练
    print('Start training ...')
    epoch = opt['epoch']
    for i in range(epoch):
        train_step(opt, model, train_iter, optimizer, ctiterion)

    # train + validation
    # valildation 时需要重写 decoder


if __name__ == '__main__':
    main()
