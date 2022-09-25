import torch.nn as nn
import torch
import statistics
    
def build_loss_seq2seq(opt):
    padding_idx = opt['vocab']['tgt_pad']
    return nn.CrossEntropyLoss(ignore_index=padding_idx, reduce='none')


# class NMTLossCompute(nn.Module):
#     def __init__(self, criterion, generator):
#         super(NMTLossCompute, self).__init__()
#         self.criterion = criterion
        
#     def _compute_loss(self, output, target):
#         '''
#         Args:
# 			output: (tgt_len, batch_size, vocab_size)
# 			target: (tgt_len, batch_size, 1)
#         '''
#         output = output.view(-1, output.size(2))
#         gtruth = target.contiguous().view(-1)
        
#         loss = self.criterion(output, gtruth)
#         stats = self._stats(loss.clone(), output, gtruth)
#         return loss, stats
    
#     def __call__(self, output, target):
#         '''
#         Function:
#             cut output and target into pieces
#             and calculate the loss independectly
#             and finally autograd(in shard)
#         '''
        
#         batch_size = output.shape[1]
#         batch_stats = statistics.Statistics()
        
#         for out_c, tgt_c in shard(output, target[1:], self.shard_size):
#             loss_step, stats = self._compute_loss(out_c, tgt_c)
#             loss_step.div(float(batch_size)).backward()
#             # 每一次 backward 都将每个节点对应 loss 的梯度累加，并清除计算图
#             batch_stats.update(stats)
#         return batch_stats
    
#     def _stats(self, loss, scores, target):
#         pred = scores.max(1)[1]
#         non_padding = target.ne(self.padding_idx)
#         num_correct = pred.eq(target).masked_select(non_padding).sum().item()
#         num_non_padding = non_padding.sum().item()
#         return statistics.Statistics(loss.item(), num_non_padding, num_correct)
        
#     @property
#     def padding_idx(self):
#         return self.criterion.ignore_index


# def filter_shard(v, shard_size):
#     v_split = []
#     for v_chunk in torch.split(v, shard_size):
#         v_chunk = v_chunk.data.clone()
#         v_chunk.requires_grad = v.requires_grad
#         v_split.append(v_chunk)
#     return v_split


# def shard(output, target, shard_size):
#     out_split = filter_shard(output, shard_size)
#     tgt_split = filter_shard(target, shard_size)
#     for p in zip(out_split, tgt_split):
#         yield p
#     torch.autograd.backward(torch.split(output, shard_size), [out_chunk.grad for out_chunk in out_split])