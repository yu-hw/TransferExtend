import torch
from torch.nn.functional import softmax

def sequence_mask(x, valid_lens, value):
    """return mask according to lengths
    Args:
        x (tensor):
            [batch_size, len]
        valid_lens (tensor):
            [batch_size]
        value (float):
    """
    shape = x.shape
    device = x.device
    
    mask = (torch.arange(0, shape[1])
            .unsqueeze(dim=1)
            .repeat(1, shape[0])
            .lt(valid_lens) # broadcast
            .permute(1, 0)
            .to(device))
    return x.masked_fill(~mask, value)


def masked_softmax(x, valid_lens):
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