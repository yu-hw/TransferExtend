import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class StackedLSTM(nn.Module):
    """
    Implacement of nn.LSTM
    Needed for the decoder, because we do input feeding.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
    
    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        
        return input, (h_1, c_1)
            
            
class GlobalAttention(nn.Module):
    """
    Attention model
    """
    def __init__(self, dim):
        raise NotImplementedError
        super(GlobalAttention, self).__init__()
        
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
    
    def score(self, h_t, h_s):
        raise NotImplementedError
        '''
        Args
            h_t: (batch_size, tgt_len, hidden_size)
            h_s: (batch_size, src_len, hidden_size)
        '''
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        h_t_ = h_t.view(tgt_batch * tgt_len, -1)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, -1)
        h_s_ = h_s.permute(0, 2, 1)
        # (batch_size, tgt_len=1, src_len)
        return torch.bmm(h_t, h_s_)
    
    def forward(self, source, memory_bank, memory_lengths):
        raise NotImplementedError
        '''
        Args:
            source: (batch_size, hidden_size)
            memory_bank: (batch_size, tgt_len, hidden_size)
            memory_lengths: (batch_size)
        '''
        source = source.unsqueeze(1)
        align = self.score(source, memory_bank) # (1, src_len)
        batch_size, tgt_len, src_len = align.size()
        
        def sequence_mask(lengths, device):
            batch_size = lengths.numel()
            max_len = lengths.max()
            return (torch.arange(0, max_len)
                    .type_as(lengths)
                    .repeat(batch_size, 1)
                    .lt(lengths.unsqueeze(1))
                    .to(device))
        
        mask = sequence_mask(memory_lengths, device=align.device)
        mask = mask.unsqueeze(1)
        align.masked_fill_(~mask, -float('inf'))
        
        align_vectors = softmax(align.view(batch_size * tgt_len, -1), -1)
        align_vectors = align_vectors.view(batch_size, tgt_len, -1)
        
        c = torch.bmm(align_vectors, memory_bank)
        
        concat_c = torch.cat([c, source], 2).view(batch_size * tgt_len, -1)
        attn_h = self.linear_out(concat_c).view(batch_size, tgt_len, -1)
        attn_h = torch.tanh(attn_h)
        
        # one_step
        attn_h = attn_h.squeeze(1)
        align_vectors = align_vectors.squeeze(1)
        
        return attn_h, align_vectors


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
        """
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
        
        # 使用 pack sequence 方式加速训练
        # 介绍可见 https://chlience.cn/2022/05/09/packed-padded-seqence-and-mask/
        embeddedSeq = self.embedding(srcTensorPadded)
        packedSeq = pack_padded_sequence(embeddedSeq, srcLengths, enforce_sorted=False)
        rnnPackedSeq, hidden_state = self.rnn(packedSeq)
        memory_bank, lengths = pad_packed_sequence(rnnPackedSeq)
        return memory_bank, lengths, hidden_state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout, **kwargs):
        """
        Args:
            vocab_size (int): Vocabulary size
            embed_size (int): Vector dim for each token
            hidden_size (int): Number of hidden cell (for one direction)
            num_layers (int): Number of layers, each layer get last layer's output as input
            dropout (float): Dropping out units 
        Addition:
        """
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        input_size = embed_size + hidden_size
        self.rnn = StackedLSTM(input_size, hidden_size, num_layers, dropout)
        self.attention = GlobalAttention(hidden_size)
    
    def init_state(self, encoder_hidden):
        def _fix_enc_hidden(hidden):
            # bid = True，从 encoder 继承 hidden_state 需要将正向反向状态拼接
            hidden = torch.cat([hidden[0:hidden.size(0):2], 
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        return tuple(_fix_enc_hidden(enc_hid) for enc_hid in encoder_hidden)
    
    def forward(self, tgt, memory_bank, memory_lengths, dec_state):
        raise NotImplementedError
        dec_outs = []
        attns = []
        
        embedded = self.embedding(tgt)
        # 和 <bos> 同时输入的初始值为全零
        input_feed = dec_state[0].new(size=(embedded.shape[1], embedded.shape[2])).zero_()
        # 单步
        for embedded_step in embedded.split(1):
            decoder_input = torch.cat([embedded_step.squeeze(0), input_feed], dim=1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            decoder_output, p_attn = self.attention(
                rnn_output,
                memory_bank.permute(1, 0, 2),
                memory_lengths)
            attns.append(p_attn)
            input_feed = decoder_output
            dec_outs += [decoder_output]
        return dec_state, dec_outs, attns