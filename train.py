import module.model as model
import module.vocab as vocab
import module.dataload as dataload
import module.dataprocess as dataprocess
import module.optimizer as optimizer
import module.iterator as iterator
import module.loss as loss
import module.statistics as statistics
import module.utils as utils

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
    # return model.buildOriginModel(opt)


def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)


def build_loss(opt):
    return loss.build_Multitask_loss(opt)


def train_step(opt, net, iterator, optimizer, ctiterion):
    device = opt['device']
    net.to(device)
    net.train()
    nmt_epoch_state = statistics.Statistics()
    mlp_epoch_state = statistics.Statistics()
    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device).permute(1, 0)
        tgt = tgt.to(device).permute(1, 0)
        label = label.to(device)
        dec_outs, mlp_outs = net(src, tgt, src_len, tgt_len)
        optimizer.zero_grad()
        loss, nmt_loss, mlp_loss, nmt_state, mlp_state= ctiterion(dec_outs, tgt, mlp_outs, label)
        loss.backward()
        # utils.clip_gradients(net, 1)
        optimizer.step()
        nmt_epoch_state.update(nmt_state)
        mlp_epoch_state.update(mlp_state)
    print(f"Train | data_num={len(iterator.dataset):6} | time={nmt_epoch_state.elapsed_time():4.1}s | nmt_acc={nmt_epoch_state.accuracy():.3f} | nmt_loss={nmt_epoch_state.xent():.3f} | mlp_acc={mlp_epoch_state.accuracy():.3f} | mlp_loss={mlp_epoch_state.xent():.3f}")
            

def validation_step(opt, net, iterator, ctiterion):
    device = opt['device']
    net.to(device)
    net.eval()
    nmt_epoch_state = statistics.Statistics()
    mlp_epoch_state = statistics.Statistics()
    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device).permute(1, 0)
        tgt = tgt.to(device).permute(1, 0)
        label = label.to(device)
        dec_outs, mlp_outs = net.validation(src, src_len, tgt, tgt.shape[0] - 1)
        loss, nmt_loss, mlp_loss, nmt_state, mlp_state = ctiterion(dec_outs, tgt, mlp_outs, label, train=False)
        nmt_epoch_state.update(nmt_state)
        mlp_epoch_state.update(mlp_state)
    print(f"Valid | data_num={len(iterator.dataset):6} | time={nmt_epoch_state.elapsed_time():4.1}s | nmt_acc={nmt_epoch_state.accuracy():.3f} | nmt_loss={nmt_epoch_state.xent():.3f} | mlp_acc={mlp_epoch_state.accuracy():.3f} | mlp_loss={mlp_epoch_state.xent():.3f}")


def main():
    print("### Load option")
    opt = load_option()
    opt['device'] = utils.get_device()
    print(opt)

    print("### Load data")
    data = load_data(opt)

    print("### Build vocabulary")
    src_vocab, tgt_vocab = build_vocab(opt, data)
    print(f"source vocab size = {len(src_vocab)}")
    print(f"target vocab size = {len(tgt_vocab)}")

    print("### Convert text to id")
    vocab.data_convert(data, src_vocab, tgt_vocab)

    print("### Add bos, eos and truncate data")
    data_process(opt, data)

    print("### Build iterator")
    print("Num of train examples:" + str(len(data['train']['source'])))
    print("Num of valid examples:" + str(len(data['valid']['source'])))
    print("Num of test  examples:" + str(len(data['test']['source'])))
    train_iter = build_iterator(opt, data['train'])
    valid_iter = build_iterator(opt, data['valid'])
    test_iter = build_iterator(opt, data['test'])

    print("### Build net")
    net = build_net(opt)
    print("模型为：", net)

    parameters = net.parameters()
    print("参数个数为：", utils.count_parameters(net))

    # 6.优化器、损失函数
    print("### Build optimizer and loss")
    optimizer = build_optimizer(opt, net)
    ctiterion = build_loss(opt)
    # 7.训练
    print('Start training ...')
    epoch = opt['epoch']
    for i in range(epoch):
        print("[Epoch " + str(i) + '/' + str(epoch) + ']')
        train_step(opt, net, train_iter, optimizer, ctiterion)
        validation_step(opt, net, valid_iter, ctiterion)
    

if __name__ == '__main__':
    main()
