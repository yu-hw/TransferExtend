
import torch

def get_opt():
    opt = {}

    opt['embed_size'] = 128
    opt['hidden_size'] = 256
    opt['dropout'] = 0.3
    opt['learning_rate'] = 0.001 # Adam
    opt['epoch'] = 10
    opt['shard_size'] = 32
    opt['NMT_weight'] = 0.2
    opt['MLP_weight'] = 0.8
    opt['max_pred_len'] = 300
    
    opt['data'] = {}
    opt['data']['fault_type'] = "MutateConditionalExpr"
    opt['data']['shuffle'] = True
    opt['data']['batch_size'] = 32

    opt['vocab'] = {}
    opt['vocab']['vocab_size'] = 30000
    opt['vocab']['share_vocab'] = True
    
    ###

    opt['encoder'] = {}
    opt['encoder']['vocab_size'] = opt['vocab']['vocab_size']
    opt['encoder']['embed_size'] = opt['embed_size']
    opt['encoder']['hidden_size'] = opt['hidden_size'] // 2
    opt['encoder']['num_layers'] = 2
    opt['encoder']['dropout'] = opt['dropout']

    opt['decoder'] = {}
    opt['decoder']['vocab_size'] = opt['vocab']['vocab_size']
    opt['decoder']['embed_size'] = opt['embed_size']
    opt['decoder']['hidden_size'] = opt['hidden_size']
    opt['decoder']['num_layers'] = 2
    opt['decoder']['dropout'] = opt['dropout']
    
    opt['mlp'] = {}
    opt['mlp']['input_size'] = opt['hidden_size']
    opt['mlp']['hidden_size'] = opt['hidden_size'] // 2
    opt['mlp']['output_size'] = 2
    opt['mlp']['dropout'] = opt['dropout']
    
    return opt