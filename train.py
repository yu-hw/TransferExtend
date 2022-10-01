import module.model as model
import module.vocab as vocab
import module.dataload as dataload
import module.dataprocess as dataprocess
import module.optimizer as optimizer
import module.iterator as iterator
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
    return dataload.load_data(opt)


def build_vocab(opt, data):
    src = data['train']['source'] + \
        data['valid']['source'] + data['test']['source']
    tgt = data['train']['target'] + \
        data['valid']['target'] + data['test']['target']
    return vocab.build_vocab(opt, src, tgt)


def data_process(opt, data):
    dataprocess.add_bos_eos(opt, data)
    dataprocess.align_data(opt, data)
    return


def build_iterator(opt, data):
    return iterator.build_iterator(opt, data)


def build_net(opt):
    return model.buildSeq2SeqModel(opt)


def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)


def build_loss(opt):
    return loss.build_loss_seq2seq(opt)


def train_step(opt, net, iterator, optimizer, ctiterion):
    device = opt['device']
    print(device)

    net.to(device)
    net.train()

    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device)
        tgt = tgt.to(device)
        label = label.to(device)
        src_len = src_len.to(device)
        tgt_len = tgt_len.to(device)
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)
        optimizer.zero_grad()
        outs = net(opt, src, tgt, src_len, tgt_len)
        gtruth = tgt[1:]
        l = ctiterion(outs.permute(1, 2, 0), gtruth.permute(1, 0))
        l.sum().backward()
        utils.clip_gradients(net, 1)
        predict_num_tokens = tgt_len.sum() - len(tgt_len)  # 去掉 <bos>
        optimizer.step()
        print("Step = " + str(i) + " " + "loss = " + str(l.sum()))
        # 补充一个统计用模块


def main():
    print("### Load option")
    opt = setting.get_opt()
    opt['device'] = utils.get_device()

    print("### Load data")
    data = load_data(opt)

    print("### Build vocabulary")
    src_vocab, tgt_vocab = build_vocab(opt, data)

    # <update>
    print("### Convert text to id")
    vocab.data_convert(data, src_vocab, tgt_vocab)

    print("### Add bos, eos and truncate data")
    data_process(opt, data)

    print("### Build iterator")
    train_iter = build_iterator(opt, data['train'])
    valid_iter = build_iterator(opt, data['valid'])
    test_iter = build_iterator(opt, data['test'])

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
        print("Epoch = " + str(i))
        train_step(opt, model, train_iter, optimizer, ctiterion)

    # train + validation
    # valildation 时需要重写 decoder


if __name__ == '__main__':
    main()
