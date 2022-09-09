import torch
from torch.nn.functional import softmax

def sequence_mask(x, valid_lens, value):
    shape = x.size()
    device = x.device
    
    mask = (torch.arange(0, shape[1])
            .repeat(shape[0], 1)
            .lt(valid_lens) # broadcast
            .to(device))
    return x.masked_fill(~mask, value)


def masked_softmax(self, x, valid_lens):
    return softmax(sequence_mask(x, valid_lens, -float('inf')))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def net_parameters(net):
    return [p for p in net.parameters() if p.requires_grad]


def data2tensor(batch, device):
    src = [torch.tensor(example[0], device=device) for example in batch]
    tgt = [torch.tensor(example[1], device=device) for example in batch]
    return src, tgt