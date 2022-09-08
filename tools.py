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