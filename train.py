import model
import setting
import utils

def load_data():
    raise NotImplementedError

def build_vocab(opt):
    opt['src_pad'] = 100
    opt['tgt_pad'] = 101
    opt['tgt_bos'] = 102
    opt['tgt_eos'] = 103
    # raise NotImplementedError

def build_iterator():
    raise NotImplementedError

def build_net():
    raise NotImplementedError

def build_optimizer():
    raise NotImplementedError

def build_loss():
    raise NotImplementedError

def train():
    raise NotImplementedError

def main():
    opt = setting.get_opt()
    opt['device'] = utils.get_device()
    build_vocab(opt['vocab'])
    
    net = model.buildMultitaskModel(opt)
    data = [[[1, 2, 3], [11, 12, 13]], [[4, 5], [14, 15]], [[6], [16]]]
    out = net(opt, data)
    print("fuck this world")
    

if __name__ == '__main__':
    main()
