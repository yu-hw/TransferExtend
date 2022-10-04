from numpy import require
import torch.nn as nn
import torch
from . import statistics


def build_loss_seq2seq(opt):
    padding_idx = opt['vocab']['tgt_pad']
    return nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')


def build_MLP_loss(opt):
    return nn.CrossEntropyLoss(reduction='sum')


def build_shard_loss(opt):
    shard_size = opt['shard_size']
    padding_idx = opt['vocab']['tgt_pad']
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
    return NMTLossCompute(criterion, shard_size)


def build_Multitask_loss(opt):
    shard_size = opt['shard_size']
    padding_idx = opt['vocab']['tgt_pad']
    NMT_weight = opt['NMT_weight']
    MLP_weight = opt['MLP_weight']
    loss1 = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
    loss2 = nn.CrossEntropyLoss(reduction='sum')
    return MultiTaskLossCompute(loss1, shard_size, loss2, NMT_weight, MLP_weight)


class NMTLossCompute(nn.Module):
    def __init__(self, criterion, shard_size):
        super(NMTLossCompute, self).__init__()
        self.criterion = criterion
        self.shard_size = shard_size
        
    def _compute_loss(self, output, target):
        '''
        Args:
			output: (shard_size, batch_size, vocab_size)
			target: (shard_size, batch_size, 1)
        '''
        output = output.view(-1, output.size(2))
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(output, gtruth)
        stats = self._stats(loss.clone(), output, gtruth)
        return loss, stats
    
    def __call__(self, output, target):
        '''
        Function:
            cut output and target into pieces
            and calculate the loss independectly
            and finally autograd(in shard)
        '''
        
        batch_size = output.shape[1]
        batch_stats = statistics.Statistics()
        
        for out_c, tgt_c in shard(output, target[1:], self.shard_size):
            loss_step, stats = self._compute_loss(out_c, tgt_c)
            loss_step.div(float(batch_size)).backward()
            # 每一次 backward 都将每个节点对应 loss 的梯度累加，并清除计算图
            batch_stats.update(stats)
        return batch_stats
    
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
    def __init__(self, NMT_criterion, shard_size, MLP_criterion, NMT_weight, MLP_weight):
        super(MultiTaskLossCompute, self).__init__()
        self.NMT_criterion = NMT_criterion
        self.shard_size = shard_size
        self.MLP_criterion = MLP_criterion
        self.NMT_weight = NMT_weight
        self.MLP_weight = MLP_weight
        
    def _NMT_compute_loss(self, output, target):
        '''
        Args:
			output: (shard_size, batch_size, vocab_size)
			target: (shard_size, batch_size, 1)
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
            cut output and target into pieces
            and calculate the loss independectly
            and finally autograd(in shard)
        '''
        batch_size = output.shape[1]
        NMT_stats = statistics.Statistics()
        MLP_stats = statistics.Statistics()
        
        loss, stats = self._MLP_compute_loss(pred, label)
        if(train):
            loss.div(float(batch_size)).mul(self.MLP_weight).backward(retain_graph=True)
        MLP_stats.update(stats)
        
        for out_c, tgt_c in shard(output, target[1:], self.shard_size, train):
            loss_step, stats = self._NMT_compute_loss(out_c, tgt_c)
            if(train):
                loss_step.div(float(tgt_c.shape[0] * tgt_c.shape[1])).backward()
            NMT_stats.update(stats)
        return NMT_stats, MLP_stats
    
    def _stats(self, loss, scores, target, padding_idx):
        pred = scores.max(1)[1]
        non_padding = target.ne(padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return statistics.Statistics(loss.item(), num_non_padding, num_correct)
    
    @property
    def padding_idx(self):
        return self.criterion.ignore_index


def filter_shard(v, shard_size):
    v_split = []
    for v_chunk in torch.split(v, shard_size):
        v_chunk = v_chunk.data.clone()
        v_chunk.requires_grad = v.requires_grad
        v_split.append(v_chunk)
    return v_split


def shard(output, target, shard_size, train=True):
    out_split = filter_shard(output, shard_size)
    tgt_split = filter_shard(target, shard_size)
    for p in zip(out_split, tgt_split):
        yield p
    if(train):
        torch.autograd.backward(torch.split(output, shard_size), [out_chunk.grad for out_chunk in out_split])