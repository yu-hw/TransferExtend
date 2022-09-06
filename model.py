import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, 
                 dropout, **kwargs):
        """
        Args:
            vocab_size (int): Vocabulary size
            embed_size (int): Vector dim for each token
            hidden_size (int): Number of hidden cell for one direction
            num_layers (int): Number of layers, each layer get last layer's output as input
            dropout (float): Dropping out units 
        Addition:
            as usual, use 'view' to seperate forward and backward
        """
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=True)
    def forward(self, src):
        """_summary_

        Args:
            src (list): list of source text
        Returns:
            memory_bank (tensor):
                (lens, batchSize, hiddenSize * 2)
            lengths (tensor): lengths of src
            hidden_state (tensor, tensor):
                ([numLayers * 2, batchSize, hiddenSize], [numLayers * 2, batchSize, hiddenSize])
        """
        srcTensorList = []
        srcLengths = []
        for line in src:
            srcLengths.append(len(line))
            srcTensorList.append(torch.tensor(line))
        srcTensorPadded = pad_sequence(srcTensorList, batch_first=True).permute(1, 0) # 将 list of tensor 填充为 tensor vec 并修改为 batch_first=False
        
        embeddedSeq = self.embedding(srcTensorPadded)
        packedSeq = pack_padded_sequence(embeddedSeq, srcLengths)
        rnnPackedSeq, hidden_state = self.rnn(packedSeq)
        memory_bank, lengths = pad_packed_sequence(rnnPackedSeq)
        return memory_bank, lengths, hidden_state

