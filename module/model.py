import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from module.utils import masked_softmax
import torch.nn.functional as F


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

    def forward(self, h, s, enc_lens):
        """Attention Step
        
        Args:
            h (tensor): Decoder arg
                (batch_size, hidden_size)
            s (tensor): Encoder outputs
                (batch_size, len, hidden_size)
        """
        h = h.unsqueeze(1).repeat((1, s.shape[1], 1))
        source = torch.cat((s, h), dim=-1)
        features = torch.tanh(self.W1(source))
        scores = self.w2(features).squeeze(-1)
        attention_weights = masked_softmax(scores, enc_lens).unsqueeze(1)
        c = torch.bmm(self.dropout(attention_weights), s)
        return c.squeeze(1), attention_weights.squeeze(1)
   

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=True)

    def forward(self, src, src_len):
        """Encoder Step
        
        Args:
            src (tensor): source
                (len, batch_size)
            src_len (tensor): source length
                (batch_size)
        Returns:
            enc_outs (tensor): Encoder outputs
                (len, batch_size, hidden_size)
            enc_lens (tensor): Encoder valid output lengths
                (batch_size)
            enc_state (tuple):
                2 * (numLayers * 2, batch_size, hidden_size)
        """
        # ?????? pack sequence ??????????????????
        # ???????????? https://chlience.cn/2022/05/09/packed-padded-seqence-and-mask/
        
        embedded_seq = self.embedding(src)
        packed_seq = pack_padded_sequence(
            embedded_seq, src_len, enforce_sorted=False)
        rnnpacked_seq, enc_state = self.rnn(packed_seq)
        enc_outs, enc_lens = pad_packed_sequence(rnnpacked_seq)
        
        # enc_outs, enc_state = self.rnn(embedded_seq)
        return enc_outs, enc_lens, enc_state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        input_size = embed_size + hidden_size
        self.rnn = StackedLSTM(input_size, hidden_size, num_layers, dropout)
        self.attention = AdditiveAttention(hidden_size, dropout)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_state):
        def _fix_enc_hidden(hidden):
            # bid = True?????? encoder ?????? hidden_state ?????????????????????????????????
            hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        return tuple(_fix_enc_hidden(enc_hid) for enc_hid in enc_state)

    def forward(self, tgt, enc_outs, enc_lens, enc_state):
        """Decoder Step with Attention

        Args:
            tgt (list of tensor): list of target tensor
            enc_outs (tensor): Encoder outputs
                (len, batch_size, hidden_size)
            enc_lens (tensor): Encoder valid output lengths
            enc_state (tuple): Encoder hidden state

        Returns:
            dec_outs (tensor): Decoder outputs
                (len, batch_size, hidden_size)
            dec_state (list): Decoder hidden state
                2 * (num_layer, batch_size, hidden_size)
            atten (tensor): attention matrix
        """
        # target ???????????? <bos> ???????????? <eos>
        # ?????? tgt[:-1] ????????? tgt[0:]
        # gtruth[0:] ??? tgt[0:] ??? <pad> ????????????????????????
        
        # decoder ??????????????? [-inf, inf]
        # ???????????????????????? softmax
        dec_outs, attns = [], []
        embedded = self.embedding(tgt[:-1])
        dec_state = self.init_state(enc_state)
        for embedded_step in embedded.split(1):
            query = dec_state[0][-1]  # dec_state[0] = h, dec_state[1] = c
            attn_c, attn_weights = self.attention(
                query,
                enc_outs.permute(1, 0, 2),
                enc_lens)
            attns.append(attn_weights)
            rnn_input = torch.cat([embedded_step.squeeze(0), attn_c], dim=1)
            dec_out, dec_state = self.rnn(rnn_input, dec_state)
            dec_outs.append(dec_out)

        dec_outs = self.dense(torch.stack(dec_outs, dim=0))
        attns = torch.stack(attns, dim=0)
        return dec_outs, dec_state, attns
    
    def validation(self, tgt, pred_max_len, enc_outs, enc_lens, enc_state):
        dec_outs, attns = [], []
        last_pred = tgt[0]
        dec_state = self.init_state(enc_state)
        while(len(dec_outs) < pred_max_len):
            embedded_step = self.embedding(last_pred)
            query = dec_state[0][-1]
            attn_c, attn_weights = self.attention(
                query,
                enc_outs.permute(1, 0, 2),
                enc_lens)
            attns.append(attn_weights)
            rnn_input = torch.cat([embedded_step, attn_c], dim=1)
            dec_out, dec_state = self.rnn(rnn_input, dec_state)
            dec_outs.append(self.dense(dec_out))
            last_pred = dec_outs[-1].argmax(-1)

        dec_outs = torch.stack(dec_outs, dim=0)
        attns = torch.stack(attns, dim=0)
        return dec_outs, dec_state, attns


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, opt, src, tgt, src_len, tgt_len):
        """
        Args:
        Returns:
            dec_outs (tensor): raw output of decoder, range in [-inf, inf]
                (lens, batch_size, vocab_size)
            attns (tensor): attention weights
                [lens, batch_size, lens]
        """
        enc_outs, enc_lens, enc_state = self.encoder(src, src_len)
        dec_outs, dec_state, attns = self.decoder(
            tgt, enc_outs, enc_lens, enc_state)
        return dec_outs

    def validation_step(self, src):
        raise NotImplementedError


class MLPModel(nn.Module):
    def __init__(self, layer_size, dropout):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential()
        layer_num = len(layer_size) - 1
        for i in range(layer_num):
            self.model.add_module('Linear ' + str(i), nn.Linear(layer_size[i], layer_size[i + 1]))
            if(i + 1 != layer_num):
                self.model.add_module('ReLU ' + str(i), nn.ReLU())
                self.model.add_module('Dropout ' + str(i), nn.Dropout(dropout))
        
    def init_input(self, enc_outs, enc_lens):
        """Got Encoder output as MLP input

        Args:
            enc_outs (tensor): Encoder output
            enc_lens (tesor): Encoder valid output lengths

        Returns:
            tensor: MLP input
                (batch_size, hidden_size)
        """
        enc_outs = enc_outs.permute(1, 2, 0)
        return F.max_pool1d(enc_outs, enc_outs.size(2)).squeeze(2)

    def forward(self, enc_outs, enc_lens):
        """MLP step

        Args:
            enc_outs (tensor): Encoder outputs
            enc_lens (tensor): Encoder valid output lengths

        Returns:
            tensor: MLP outputs
        """
        input = self.init_input(enc_outs, enc_lens)
        out = self.model(input)
        return out


class MultitaskModel(nn.Module):
    def __init__(self, encoder, decoder, mlp):
        super(MultitaskModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp = mlp

    def forward(self, src, tgt, src_len, tgt_len):
        """MultitaskModel Train One Step

        Args:
            opt (dict): _description_
            src (tensor): source
                (len, batch_size, hidden_size)
            tgt (tensor): target
                (len, batch_size, hidden_size)
            src_len (tensor): source length
                (batch_size)
            tgt_len (tensor): target length
                (batch_size)

        Returns:
            tensor: value in range [-inf, inf]
                (lens, batch_size, vocab_size)
            tensor: value in range [-inf, inf] 
                (batch_size)
            tensor: attention weights
        """
        enc_outs, enc_lens, enc_state = self.encoder(src, src_len)
        dec_outs, dec_state, attns = self.decoder(
            tgt, enc_outs, enc_lens, enc_state)
        mlp_out = self.mlp(enc_outs, enc_lens)
        return dec_outs, mlp_out

    def validation(self, src, src_len, tgt, pred_max_len):
        enc_outs, enc_lens, enc_state = self.encoder(src, src_len)
        dec_outs, dec_state, attns = self.decoder.validation(
            tgt, pred_max_len, enc_outs, enc_lens, enc_state)
        mlp_out = self.mlp(enc_outs, enc_lens)
        return dec_outs, mlp_out
    

class OriginModel(nn.Module):
    def __init__(self, encoder, mlp):
        super(OriginModel, self).__init__()
        self.encoder = encoder
        self.mlp = mlp

    def forward(self, src, src_len):
        """MultitaskModel Train One Step

        Args:
            opt (dict): _description_
            src (tensor): source
                (len, batch_size, hidden_size)
            src_len (tensor): source length
                (batch_size)

        Returns:
            tensor: value in range [-inf, inf] 
                (batch_size)
            tensor: attention weights
        """
        enc_outs, enc_lens, enc_state = self.encoder(src, src_len)
        mlp_out = self.mlp(enc_outs, enc_lens)
        return mlp_out

    def validation(self, src, src_len, tgt_bos, pred_max_len):
        enc_outs, enc_lens, enc_state = self.encoder(src, src_len)
        mlp_out = self.mlp(enc_outs, enc_lens)
        return mlp_out


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
    layer_size = opt['layer_size']
    dropout = opt['dropout']
    return MLPModel(layer_size, dropout)


def buildSeq2SeqModel(opt):
    opt['encoder']['vocab_size'] = opt['vocab']['src_vocab_size']
    opt['decoder']['vocab_size'] = opt['vocab']['tgt_vocab_size']
    encoder = buildEncoder(opt['encoder'])
    decoder = buildDecoder(opt['decoder'])
    return Seq2SeqModel(encoder, decoder)


def buildMultitaskModel(opt):
    opt['encoder']['vocab_size'] = opt['vocab']['src_vocab_size']
    opt['decoder']['vocab_size'] = opt['vocab']['tgt_vocab_size']
    encoder = buildEncoder(opt['encoder'])
    decoder = buildDecoder(opt['decoder'])
    mlp = buildMLP(opt['mlp'])
    return MultitaskModel(encoder, decoder, mlp)


def buildOriginModel(opt):
    opt['encoder']['vocab_size'] = opt['vocab']['src_vocab_size']
    encoder = buildEncoder(opt['encoder'])
    mlp = buildMLP(opt['mlp'])
    return OriginModel(encoder, mlp)