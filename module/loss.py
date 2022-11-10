from numpy import require
import torch.nn as nn
import statistics


def build_loss_seq2seq(opt):
    padding_idx = opt['vocab']['tgt_pad']
    return nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')


def build_MLP_loss(opt):
    return nn.CrossEntropyLoss(reduction='sum')


def build_shard_loss(opt):
    padding_idx = opt['vocab']['tgt_pad']
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
    return NMTLossCompute(criterion)


def build_Multitask_loss(opt):
    padding_idx = opt['vocab']['tgt_pad']
    NMT_weight = opt['NMT_weight']
    MLP_weight = opt['MLP_weight']
    loss1 = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
    loss2 = nn.CrossEntropyLoss(reduction='sum')
    return MultiTaskLossCompute(loss1, loss2, NMT_weight, MLP_weight)


class NMTLossCompute(nn.Module):
    def __init__(self, criterion):
        super(NMTLossCompute, self).__init__()
        self.criterion = criterion
        
    def _compute_loss(self, output, target):
        '''
        Args:
			output: (len, batch_size, vocab_size)
			target: (len, batch_size, 1)
        '''
        output = output.view(-1, output.size(2))
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(output, gtruth)
        stats = self._stats(loss.clone(), output, gtruth)
        return loss, stats
    
    def __call__(self, output, target):
        '''
        Function:
            calculate the loss independectly
        '''
        loss, stats = self._compute_loss(output, target[1:])
        loss = loss.div(float(stats.n_words))
        
        return loss, stats
    
    def _stats(self, loss, scores, target):
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return statistics.Statistics(loss.item(), num_non_padding, num_correct)
        
    @property
    def padding_idx(self):
        return self.criterion.ignore_index


class MultiTaskLossCompute(nn.Module):
    def __init__(self, NMT_criterion, MLP_criterion, NMT_weight, MLP_weight):
        super(MultiTaskLossCompute, self).__init__()
        self.NMT_criterion = NMT_criterion
        self.MLP_criterion = MLP_criterion
        self.NMT_weight = NMT_weight
        self.MLP_weight = MLP_weight
        
    def _NMT_compute_loss(self, output, target):
        '''
        Args:
			output: (len, batch_size, vocab_size)
			target: (len, batch_size, 1)
        '''
        output = output.view(-1, output.size(2))
        gtruth = target.contiguous().view(-1)
        loss = self.NMT_criterion(output, gtruth)
        stats = self._stats(loss.clone(), output, gtruth, self.NMT_criterion.ignore_index)
        return loss, stats
    
    def _MLP_compute_loss(self, output, target):
        gtruth = target.contiguous()
        loss = self.MLP_criterion(output, gtruth)
        stats = self._stats(loss.clone(), output, gtruth, self.MLP_criterion.ignore_index)
        return loss, stats
    
    def __call__(self, output, target, pred, label, train=True):
        '''
        Function:
            calculate the loss independectly
        '''
        MLP_loss, MLP_stats = self._MLP_compute_loss(pred, label)
        NMT_loss, NMT_stats = self._NMT_compute_loss(output, target[1:])
        
        MLP_loss = MLP_loss.div(float(MLP_stats.n_words))
        NMT_loss = NMT_loss.div(float(NMT_stats.n_words))
        
        loss = MLP_loss * self.MLP_weight + NMT_loss * self.NMT_weight
        return loss, NMT_loss, MLP_loss, NMT_stats, MLP_stats
    
    def _stats(self, loss, scores, target, padding_idx):
        pred = scores.max(1)[1]
        non_padding = target.ne(padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return statistics.Statistics(loss.item(), num_non_padding, num_correct)
    
    @property
    def padding_idx(self):
        return self.criterion.ignore_index