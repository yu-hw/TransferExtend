import module.model as model
import module.vocab as vocab
import module.dataload as dataload
import module.dataprocess as dataprocess
import module.optimizer as optimizer
import module.iterator as iterator
import module.loss as loss
import module.statistics as statistics
import setting
import utils


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
    return loss.build_shard_loss(opt)


def train_step(opt, net, iterator, optimizer, ctiterion):
    device = opt['device']
    print('Training device:' + str(device))

    net.to(device)
    net.train()
    
    epoch_stats = statistics.Statistics()
    for i, data in enumerate(iterator):
        src, tgt, label, src_len, tgt_len = data
        src = src.to(device)
        tgt = tgt.to(device)
        label = label.to(device)
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)
        optimizer.zero_grad()
        outs = net(opt, src, tgt, src_len, tgt_len)
        batch_stats = ctiterion(outs, tgt)
        epoch_stats.update(batch_stats)
        # utils.clip_gradients(net, 1)
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"batch: {i + 1:5} | acc={batch_stats.accuracy():.3f} | loss={batch_stats.xent():.3f} | time={epoch_stats.elapsed_time():.1f}s")


def main():
    print("### Load option")
    opt = setting.get_opt()
    opt['device'] = utils.get_device()

    print("### Load data")
    data = load_data(opt)

    print("### Build vocabulary")
    src_vocab, tgt_vocab = build_vocab(opt, data)

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

    # train + validation
    # valildation 时需要重写 decoder


if __name__ == '__main__':
    main()
