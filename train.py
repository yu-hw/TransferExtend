import module.model as model
import module.vocab as vocab
import module.dataload as dataload
import module.dataprocess as dataprocess
import module.optimizer as optimizer
import module.iterator as iterator
import module.loss as loss
import module.statistics as statistics
import module.utils

import sys
import json

def load_option():
    try:
        file_name = sys.argv[1]
        with open(file_name, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(e)
        raise Exception("Load Option Error")


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


def build_iterator(opt, data):
    return iterator.build_iterator(opt, data)


def build_net(opt):
    return model.buildMultitaskModel(opt)


def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)


def build_loss(opt):
    return loss.build_Multitask_loss(opt)


def train_step(opt, net, iterator, optimizer, ctiterion):
    device = opt['device']
    print('Training device:' + str(device))
    net.to(device)
    net.train()
    epoch_NMT_stats = statistics.Statistics()
    epoch_MLP_stats = statistics.Statistics()
    print("Train example: " + str(len(iterator.dataset)))
    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device).permute(1, 0)
        tgt = tgt.to(device).permute(1, 0)
        label = label.to(device)
        optimizer.zero_grad()
        outs, pred = net(src, tgt, src_len, tgt_len)
        loss, NMT_loss, MLP_loss, NMT_stats, MLP_stats = ctiterion(outs, tgt, pred, label)
        
        loss.backward()
        # utils.clip_gradients(net, 1)
        optimizer.step()
        
        epoch_NMT_stats.update(NMT_stats)
        epoch_MLP_stats.update(MLP_stats)
        if (i + 1) % 50 == 0:
            print(f"batch: {i + 1:5} | NMT_acc={epoch_NMT_stats.accuracy():.3f} | NMT_loss={epoch_NMT_stats.xent():.3f} | MLP_acc={epoch_MLP_stats.accuracy():.3f} | MLP_loss={epoch_MLP_stats.xent():.3f} | time={epoch_NMT_stats.elapsed_time():.1f}s")
            

def validation_step(opt, net, iterator, ctiterion):
    device = opt['device']
    print('Validation device:' + str(device))
    net.to(device)
    net.eval()
    epoch_NMT_stats = statistics.Statistics()
    epoch_MLP_stats = statistics.Statistics()
    print("Validtion example: " + str(len(iterator.dataset)))
    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device).permute(1, 0)
        tgt = tgt.to(device).permute(1, 0)
        label = label.to(device)
        outs, pred = net.validation(src, src_len, tgt[0], tgt.shape[0] - 1)
        loss, NMT_loss, MLP_loss, NMT_stats, MLP_stats = ctiterion(outs, tgt, pred, label, train=False)
        epoch_NMT_stats.update(NMT_stats)
        epoch_MLP_stats.update(MLP_stats)
    print(f"batch: {i + 1:5} | NMT_acc={epoch_NMT_stats.accuracy():.3f} | NMT_loss={epoch_NMT_stats.xent():.3f} | MLP_acc={epoch_MLP_stats.accuracy():.3f} | MLP_loss={epoch_MLP_stats.xent():.3f} | time={epoch_NMT_stats.elapsed_time():.1f}s")
        
        
def main():
    print("### Load option")
    opt = load_option()
    opt['device'] = utils.get_device()
    print(opt)

    print("### Load data")
    data = load_data(opt)

    print("### Build vocabulary")
    src_vocab, tgt_vocab = build_vocab(opt, data)

    print("### Convert text to id")
    vocab.data_convert(data, src_vocab, tgt_vocab)

    print("### Add bos, eos and truncate data")
    data_process(opt, data)

    print("### Build iterator")
    print("Num of train examples:" + str(len(data['train'])))
    print("Num of valid examples:" + str(len(data['valid'])))
    print("Num of test examples:" + str(len(data['test'])))
    train_iter = build_iterator(opt, data['train']['src'])
    valid_iter = build_iterator(opt, data['valid']['src'])
    test_iter = build_iterator(opt, data['test']['src'])

    print("### Build net")
    model = build_net(opt)
    print("模型为：", model)

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
        validation_step(opt, model, valid_iter, ctiterion)

if __name__ == '__main__':
    main()