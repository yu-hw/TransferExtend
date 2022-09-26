import torch

def build_optimizer(opt, net):
    lr = opt["learning_rate"]
    return torch.optim.Adam(net.parameters(), lr=lr)