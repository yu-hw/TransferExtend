import torch
import torch.nn as nn

class NMTLossCompute(nn.Module):
    def __init__(self, criterion, generator):
        super(NMTLossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
    
    def _compute_loss(self, output, target):
        output = output.view(-1, output.shape[-1])
        scores = self.generator(output)
        