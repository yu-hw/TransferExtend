import model
import setting
import utils
import optimizer

def load_data():
    raise NotImplementedError

def build_vocab(opt):
    raise NotImplementedError

def build_iterator():
    raise NotImplementedError

def build_net(opt):
    return model.buildSeq2SeqModel(opt)

def build_optimizer(opt, net):
    return optimizer.build_optimizer(opt, net)

def build_loss():
    raise NotImplementedError

def train():
    raise NotImplementedError

def main():
    raise NotImplementedError

if __name__ == '__main__':
    main()
