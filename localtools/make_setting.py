import json
import os
import sys

def origin_opt():
    opt = {}
    opt['embed_size'] = 32
    opt['hidden_size'] = 50
    opt['lstm_dropout'] = 0
    opt['learning_rate'] = 0.001 # Adam
    opt['epoch'] = 30
    opt['NMT_weight'] = 0
    opt['MLP_weight'] = 1
    opt['max_pred_len'] = 500
    opt['batch_size'] = 32
    opt['vocab_min_freq'] = 5
    opt['data_path'] = ""
    opt['mlp_dropout'] = 0
    opt['mlp_layer_size'] = [opt['hidden_size'] * 2, 2]
    return opt

def process_opt(opt):
    opt['data'] = {}
    opt['data']['path'] = opt['data_path']
    opt['data']['shuffle'] = True
    opt['data']['batch_size'] = opt['batch_size']

    opt['vocab'] = {}
    opt['vocab']['min_freq'] = opt['vocab_min_freq']
    opt['vocab']['max_size'] = 30000
    opt['vocab']['share_vocab'] = True

    opt['encoder'] = {}
    opt['encoder']['embed_size'] = opt['embed_size']
    opt['encoder']['hidden_size'] = opt['hidden_size']
    opt['encoder']['num_layers'] = 1
    opt['encoder']['dropout'] = opt['lstm_dropout']

    opt['decoder'] = {}
    opt['decoder']['embed_size'] = opt['embed_size']
    opt['decoder']['hidden_size'] = opt['hidden_size'] * 2
    opt['decoder']['num_layers'] = 1
    opt['decoder']['dropout'] = opt['lstm_dropout']

    opt['mlp'] = {}
    opt['mlp']['layer_size'] = opt['mlp_layer_size']
    opt['mlp']['dropout'] = opt['mlp_dropout']


def single():
    faultType = sys.argv[1]
    json_path = '/home/LAB/caohl/TransferExtend/train-setting'
    opt = origin_opt()
    opt['fault_type'] = faultType
    opt['data_path'] = '/home/LAB/caohl/TransferExtend/data/data-diff'
    process_opt(opt)
    print(f"[{faultType}]")
    print(f"{os.path.join(json_path, faultType + '.json')}")
    with open(os.path.join(json_path, faultType + '.json'), 'w') as f:
        f.write(json.dumps(opt))


def multi():
    dir_names = []
    data_path = '/home/LAB/caohl/TransferExtend/data/data-diff'
    for root, dirs, files in os.walk(data_path):
        if 'train.pkl' in files:
            dir_name = os.path.split(root)[-1]
            dir_names.append(dir_name)
    
    json_path = '/home/LAB/caohl/TransferExtend/train-setting'
    for dir_name in dir_names:
        opt = origin_opt()
        opt['data_path'] = os.path.join(data_path, dir_name)
        opt['NMT_weight'] = 0.1
        opt['MLP_weight'] = 0.9
        process_opt(opt)
        print(f"[{dir_name}]")
        print(f"{os.path.join(json_path, dir_name + '.json')}")
        with open(os.path.join(json_path, dir_name + '.json'), 'w') as f:
            f.write(json.dumps(opt))
    

if __name__ == '__main__':
    multi()