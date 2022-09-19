import module.model as model
import setting
import utils
import module.optimizer as optimizer

#  1、train文件中需要将每一个功能都分开在不同的模块文件中   2、需要修改数据集，总共有三种数据 两个标签  3、需要设置训>练集、验证集、测试集。



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
