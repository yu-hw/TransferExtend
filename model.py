from doctest import master
from xml.sax.handler import feature_string_interning
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import softmax

from tools import masked_softmax

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
            

class AdditiveAttention(nn.module):
    def __init__(self, dim, dropout):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(dim * 2, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, h, s, valid_lens):
        """
        Args:
            h (tensor): from decoder
                [batch_size, feature_dim]
            s (tensor): from encoder
                [batch_size, len, feature_dim]
        """
        h = h.unsqueeze(1).repeat((1, s.size[1], 1))
        source = torch.cat((s, h), dim=-1)
        features = torch.tanh(self.W1(source))
        scores = self.w2(features).unqueeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        c = torch.bmm(self.dropout(attention_weights), s)
        return c, attention_weights
        

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
            src (list of tensor): list of source tensor
        Returns:
            memory_bank (tensor):
                (lens, batchSize, hiddenSize * 2)
            lengths (tensor): lengths of src
            hidden_state (tensor, tensor):
                ([numLayers * 2, batchSize, hiddenSize], [numLayers * 2, batchSize, hiddenSize])
        """
        # 使用 pack sequence 方式加速训练
        # 介绍可见 https://chlience.cn/2022/05/09/packed-padded-seqence-and-mask/
        src_lengths = []
        for t in src:
            src_lengths.append(len(t))
        embedded_seq = self.embedding(src)
        packed_seq = pack_padded_sequence(embedded_seq, src_lengths, enforce_sorted=False)
        rnnpacked_seq, hidden_state = self.rnn(packed_seq)
        memory_bank, lengths = pad_packed_sequence(rnnpacked_seq)
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
        self.attention = AdditiveAttention(hidden_size, dropout)
    
    def init_state(self, encoder_hidden):
        def _fix_enc_hidden(hidden):
            # bid = True，从 encoder 继承 hidden_state 需要将正向反向状态拼接
            hidden = torch.cat([hidden[0:hidden.size(0):2], 
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        return tuple(_fix_enc_hidden(enc_hid) for enc_hid in encoder_hidden)
    
    def forward(self, opt, tgt, memory_bank, valid_lens, dec_state):
        """
        Args:
            tgt (list of tensor): list of target tensor
            memory_bank (tensor): from encoder
                (lens, batchSize, hiddenSize * 2)
            valid_lens (tensor): from encoder, lengths of src
            dec_state: hiddne_state of decoder
        """
        vocab_end = opt['vocab']['tgt_end']
        vocab_bos = opt['vocab']['tgt_bos']
        
        tgt_lengths = []
        for t in tgt:
            tgt_lengths.append(len(t))
        tgt_padded = pad_sequence(tgt, batch_first=True, padding_value=vocab_end).permute(1, 0) # 将 list of tensor 填充为 tensor vec 并修改为 batch_first=False
        bos_tensor = tgt_padded.new_full((1, tgt_padded.size[1]), vocab_bos)
        embedded = self.embedding(torch.cat((bos_tensor, tgt_padded[:-1]), dim = 0))
        
        dec_outs = []
        attns = []
        
        (h, c) = dec_state
        dec_out = h[-1].squeeze(0)
        
        for embedded_step in embedded.split(1):
            attn_c, attn_weights = self.attention(
                dec_out,
                memory_bank.permute(1, 0, 2),
                valid_lens)
            attns.append(attn_weights)
            rnn_input = torch.cat([embedded_step.squeeze(0), attn_c], dim=1)
            dec_out, dec_state = self.rnn(rnn_input, dec_state)
            dec_outs.append(dec_out)
            
        return dec_outs, dec_state , attns
    
    
class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def data2tensor(opt, batch):
        device = opt['device']
        
        src = [torch.tensor(example[0], device=device) for example in batch]
        tgt = [torch.tensor(example[1], device=device) for example in batch]
        return src, tgt
        
    def forward(self, opt, batch):
        src, tgt = self.data2tensor(opt, batch)
        enc_outs, lengths, enc_state = self.encoder(src)
        dec_state = self.decoder.init_state(enc_state)
        dec_outs, dec_state, attns = self.decoder(opt, tgt, enc_outs, lengths, dec_state)
        dec_outs = torch.stack(dec_outs)
        attns = torch.stack(attns)
        return dec_outs, attns
    