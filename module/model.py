from re import M
from turtle import forward
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import softmax
from utils import masked_softmax, data2tensor

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
            

class AdditiveAttention(nn.Module):
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
        h = h.unsqueeze(1).repeat((1, s.shape[1], 1))
        source = torch.cat((s, h), dim=-1)
        features = torch.tanh(self.W1(source))
        scores = self.w2(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens).unsqueeze(1)
        c = torch.bmm(self.dropout(attention_weights), s)
        return c.squeeze(1), attention_weights.squeeze(1)
        

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
    
    def dataProcess(self, lines, padding_value=0):
        lengths = []
        for line in lines:
            lengths.append(len(line))
        return pad_sequence(lines, padding_value=padding_value), lengths
    
    def forward(self, opt, src):
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
        vocab_pad = opt['vocab']['src_pad']
        
        src, src_lengths = self.dataProcess(src, vocab_pad)
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
        """
        super(Decoder, self).__init__(**kwargs)
        input_size = embed_size + hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = StackedLSTM(input_size, hidden_size, num_layers, dropout)
        self.attention = AdditiveAttention(hidden_size, dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)
    
    def init_state(self, encoder_hidden):
        def _fix_enc_hidden(hidden):
            # bid = True，从 encoder 继承 hidden_state 需要将正向反向状态拼接
            hidden = torch.cat([hidden[0:hidden.size(0):2], 
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        return tuple(_fix_enc_hidden(enc_hid) for enc_hid in encoder_hidden)
        
    def dataProcess(self, lines, padding_value=0):
        lengths = []
        for line in lines:
            lengths.append(len(line))
        return pad_sequence(lines, padding_value=padding_value), lengths
    
    def forward(self, opt, tgt, memory_bank, valid_lens, enc_state):
        """
        Args:
            tgt (list of tensor): list of target tensor
            memory_bank (tensor): from encoder
                (lens, batchSize, hiddenSize * 2)
            valid_lens (tensor): from encoder, lengths of src
            dec_state: hiddne_state of decoder
        """
        vocab_bos = opt['vocab']['tgt_bos']
        vocab_pad = opt['vocab']['tgt_pad']
        
        tgt, tgt_lengths = self.dataProcess(tgt, vocab_pad)
        bos_tensor = tgt.new_full((1, tgt.shape[1]), vocab_bos)
        embedded = self.embedding(torch.cat((bos_tensor, tgt[:-1]), dim = 0))
        
        dec_outs, attns = [], []
        
        dec_state = self.init_state(enc_state)
        
        for embedded_step in embedded.split(1):
            query = dec_state[0][-1] # dec_state[0] = h, dec_state[1] = c
            attn_c, attn_weights = self.attention(
                query,
                memory_bank.permute(1, 0, 2),
                valid_lens)
            attns.append(attn_weights)
            rnn_input = torch.cat([embedded_step.squeeze(0), attn_c], dim=1)
            dec_out, dec_state = self.rnn(rnn_input, dec_state)
            dec_outs.append(dec_out)
            
        dec_outs = self.dense(torch.stack(dec_outs, dim=0))
        attns = torch.stack(attns, dim=0)
        return dec_outs, dec_state , attns
    

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, opt, batch):
        """
        Args:
            opt (dict): 
            batch (list): 详见 data2tensor 模块

        Returns:
            dec_outs (tensor): raw output of decoder, range in [-inf, inf]
                (lens, batch_size, vocab_size)
            attns (tensor): attention weights
                [lens, batch_size, lens]
        """
        src, tgt = data2tensor(batch, self.device)
        enc_outs, lengths, enc_state = self.encoder(src)
        dec_outs, dec_state, attns = self.decoder(opt, tgt, enc_outs, lengths, enc_state)
        return dec_outs, attns
    
    def validation_step(self, src):
        raise NotImplementedError


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def init_input(self, enc_outs, enc_lengths):
        input = []
        for (id, pos) in enumerate(enc_lengths):
            input.append(enc_outs[pos - 1][id])
        return torch.stack(input, dim=0)
    
    def forward(self, enc_outs, enc_lengths):
        input = self.init_input(enc_outs, enc_lengths)
        out = self.fc1(input)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


class MultitaskModel(nn.Module):
    def __init__(self, encoder, decoder, mlp, device):
        super(MultitaskModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp = mlp
        self.device = device
    
    def forward(self, opt, batch):
        """
        Args:
            opt (dict): 
            batch (list): 详见 data2tensor 模块

        Returns:
            dec_outs (tensor): value in range [-inf, inf]
                (lens, batch_size, vocab_size)
            mlp_out (tensor): value in range [-inf, inf] 
                (batch_size)
            attns (tensor): attention weights
        """
        src, tgt = data2tensor(batch, self.device)
        enc_outs, lengths, enc_state = self.encoder(src)
        dec_outs, dec_state, attns = self.decoder(opt, tgt, enc_outs, lengths, enc_state)
        mlp_out = self.mlp(enc_outs, lengths)
        
        return dec_outs, mlp_out, attns
    
    def validation_step(self, src):
        raise NotImplementedError
        

def buildEncoder(opt):
    vocab_size = opt['vocab_size']
    embed_size = opt['embed_size']
    hidden_size = opt['hidden_size']
    num_layers = opt['num_layers']
    dropout = opt['dropout']
    return Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)


def buildDecoder(opt):
    vocab_size = opt['vocab_size']
    embed_size = opt['embed_size']
    hidden_size = opt['hidden_size']
    num_layers = opt['num_layers']
    dropout = opt['dropout']
    return Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)


def buildMLP(opt):
    input_size = opt['input_size']
    hidden_size = opt['hidden_size']
    output_size = opt['output_size']
    dropout = opt['dropout']
    return MLPModel(input_size, hidden_size, output_size, dropout)


def buildSeq2SeqModel(opt):
    device = opt['device']
    encoder = buildEncoder(opt['encoder'])
    decoder = buildDecoder(opt['decoder'])
    return Seq2SeqModel(encoder, decoder, device)


def buildMultitaskModel(opt):
    device = opt['device']
    encoder = buildEncoder(opt['encoder'])
    decoder = buildDecoder(opt['decoder'])
    mlp = buildMLP(opt['mlp'])
    return MultitaskModel(encoder, decoder, mlp, device)