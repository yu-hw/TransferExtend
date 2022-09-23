import os
import sys
import torch
import time
import pickle
import random
import collections
import math

from torch import nn
from torch.utils import data
import module.model as model
import setting
import utils
import module.optimizer as optimizer
from setting import get_opt
from module.vocab import Vocab
# from d2l import torch as d2l

#  1、train文件中需要将每一个功能都分开在不同的模块文件中 √
#  2、需要修改数据集，总共有三种数据 两个标签 neg数据中的rank关键字删去，生成 source、target、label三个数据 √
#  3、需要设置训练集、验证集、测试集。 √
#  4、整理代码规范 和 生成11个文件中的 train\val\test √
#  5、制作 train、val、test的数据 迭代器 √
#  6、写loss函数
#  7、将数据输入到 模型中，计算损失

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def write_pkl(obj, filepath):
    f = open(filepath,"wb")
    pickle.dump(obj,f)
    f.close()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def load_data(source_path,target_path): #返回两个词元列表
    # 1.数据加载并词元化
    source, target = [], []
    pkl_source = read_pkl(source_path)
    pkl_target = read_pkl(target_path)
    for line in pkl_source:
        source.append(list(line.split(' ')))

    for line in pkl_target:
            target.append(list(line.split(' ')))
    return source, target

def build_vocab(opt, src, tgt):  #返回构建的两个词表
    vocab_size = opt['vocab']['vocab_size']
    if opt['vocab']['share_vocab'] is True:
        src_vocab = tgt_vocab = Vocab(src + tgt, vocab_size=vocab_size, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    else:
        src_vocab = Vocab(src, vocab_size=vocab_size, reserved_tokens=['<pad>'])
        tgt_vocab = Vocab(tgt, vocab_size=vocab_size, reserved_tokens=['<pad>', '<bos>'])

    opt['src_pad'] = src_vocab['<pad>']
    opt['tgt_bos'] = tgt_vocab['<bos>']
    opt['tgt_pad'] = tgt_vocab['<pad>']
    opt['src_vocab_size'] = len(src_vocab.idx2token)
    opt['tgt_vocab_size'] = len(tgt_vocab.idx2token)
    return src_vocab, tgt_vocab

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def load_array_iter(data_arrays, batch_size, is_train=True): # 返回batch_size个样本
    # 将传入的特征和标签作为list传到TensorDataset里面得到一个pytorch的数据集（dataset）
    dataset = data.TensorDataset(*data_arrays)
    # 调用Dataloader每次从dataset里面挑选batch_size个样本出来（shuffle：是否随机）
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def build_iterator(source_lines,src_vocab,target_lines,tgt_vocab,opt,label):#返回数据的迭代器
    #1.读取数据的每一行, src_vocab[l]表示 source_lines中的每一行，再加入到source_lines中,字符转成数字
    source_lines = [src_vocab[l] for l in source_lines]
    target_lines = [tgt_vocab[l] for l in target_lines]
    #2.按照指定长度截取数据并存到array
    source_array = torch.tensor([truncate_pad(
        l, opt['num_steps'] , src_vocab['<pad>']) for l in source_lines])
    target_array = torch.tensor([truncate_pad(
        l, opt['num_steps'], tgt_vocab['<pad>']) for l in target_lines])
    label_array = torch.tensor(label)
    #3.保存每一句的有效长度
    source_valid_len = (source_array != src_vocab['<pad>']).type(torch.int32).sum(1)
    target_valid_len = (target_array != tgt_vocab['<pad>']).type(torch.int32).sum(1)
    #4.所有有效数据放在一个array中, 包括每一条数据的label    5个数据的第一维必须相等
    data_arrays = (source_array,source_valid_len,target_array,target_valid_len,label_array)
    #5.按照batch_size 构建数据的迭代器
    print("data_arrays中各个数据的大小是否一致：",data_arrays[0].size(),data_arrays[1].size(),data_arrays[2].size(),data_arrays[3].size(),data_arrays[4].size())
    data_iter = load_array_iter(data_arrays, opt['batch_size'])

    return data_iter

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
    #1. 获取划分好的 11*3*3 的数据集，每次训练一类错误类型
    #     获取数据的路径
    fault_type = "MoveStmt"
    print("fault_type: {}".format(fault_type))
    root = "./data/{}/".format(fault_type)

    train_source_path = (os.path.join(root, "train/source.pkl"))
    train_target_path = (os.path.join(root, "train/target.pkl"))
    train_label_path = (os.path.join(root, "train/label.pkl"))

    val_source_path = (os.path.join(root, "val/source.pkl"))
    val_target_path = (os.path.join(root, "val/target.pkl"))
    val_label_path = (os.path.join(root, "val/label.pkl"))

    test_source_path = (os.path.join(root, "test/source.pkl"))
    test_target_path = (os.path.join(root, "test/target.pkl"))
    test_label_path = (os.path.join(root, "test/label.pkl"))

    # 2.按照路径加载数据 返回的是 两个词元化处理的list  + 定义用于MLP训练的label数据
    # train_source,train_target 需要变成词元列表  label是list不变

    train_source,train_target = load_data(train_source_path,train_target_path)
    train_label = read_pkl(train_label_path)

    val_source, val_target = load_data(val_source_path, val_target_path)
    val_label = read_pkl(val_label_path)

    test_source, test_target = load_data(test_source_path, test_target_path)
    test_label = read_pkl(test_label_path)

    # 3.获取所有数据的词表，词表大小为30000
    opt = get_opt()
    device = try_gpu()
    total_soruce = train_source+val_source+test_source
    total_target = train_target+val_target+test_target
    src_vocab, tgt_vocab = build_vocab(opt, total_soruce,total_target )

    #4.按照batch_size，构造数据迭代器
    # iteration 返回五个数据, source,source_len,target,target_len,label
    train_iter = build_iterator(train_source,src_vocab,train_target,tgt_vocab,opt,train_label)
    val_iter = build_iterator(val_source, src_vocab, val_target, tgt_vocab, opt, val_label)
    test_iter = build_iterator(test_source, src_vocab, test_target, tgt_vocab, opt, test_label)

    #5.引用网络
    model = build_net(opt)
    print("使用的模型为：",model)
    parameters = model.parameters()
    print("模型的参数为：",model.parameters())
    
    
    #6.优化器、损失函数
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    #7.训练
    print('Start training ...')
    model.to(device)
    model.train()

    for epoch in range( opt['EPOCHS']):
        for batch in train_iter:
            # print("epoch为：",epoch,"batch为：",batch[0].size(),batch[1].size(),batch[2].size(),batch[3].size(),batch[4].size())
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len,label = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
    #         Y_hat, _ = model(X, dec_input, X_valid_len)
    #         l = loss(Y_hat, Y, Y_valid_len)
    #         l.sum().backward()  # 损失函数的标量进行“反向传播”
    #         d2l.grad_clipping(model, 1)
    #         num_tokens = Y_valid_len.sum()
    #         optimizer.step()
    #         with torch.no_grad():
    #             metric.add(l.sum(), num_tokens)
    #     if (epoch + 1) % 10 == 0:
    #         animator.add(epoch + 1, (metric[0] / metric[1],))
    # print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
    #         f'tokens/sec on {str(device)}')


if __name__ == '__main__':
    main()
