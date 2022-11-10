import json
import os

def origin_opt():
    opt = {}
    opt['embed_size'] = 32
    opt['hidden_size'] = 50
    opt['lstm_dropout'] = 0
    opt['learning_rate'] = 0.001 # Adam
    opt['epoch'] = 40
    opt['NMT_weight'] = 0
    opt['MLP_weight'] = 1
    opt['max_pred_len'] = 500
    opt['batch_size'] = 24
    opt['data_path'] = '/home/LAB/caohl/TransferExtend-new/data'
    opt['fault_type'] = "MutateDataType"
    opt['vocab_min_freq'] = 5
    
    opt['mlp_dropout'] = 0
    opt['mlp_layer_size'] = [opt['hidden_size'] * 2, 2]
    return opt

def process_opt(opt):
    opt['data'] = {}
    opt['data']['path'] = os.path.join(opt['data_path'], opt['fault_type'])
    opt['data']['fault_type'] = opt['fault_type']
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

    
def main():
    # fault_type = []
    # data_path = '/home/LAB/caohl/TransferExtend/data'
    # for root, dirs, files in os.walk(data_path):
    #     if 'train.pkl' in files:
    #         name = os.path.split(root)[-1]
    #         fault_type.append(name)
    
    json_path = '/home/LAB/caohl/TransferExtend/train-setting'
    # for name in fault_type:
    #     opt = origin_opt()
    #     opt['fault_type'] = name
    #     process_opt(opt)
    #     with open(os.path.join(json_path, name + '.json'), 'w') as f:
    #         f.write(json.dumps(opt))
    fault_type = "MutateDataType"
    opt = origin_opt()
    opt['fault_type'] = fault_type
    process_opt(opt)
    with open(os.path.join(json_path, fault_type + '.json'), 'w') as f:
        f.write(json.dumps(opt))
    

if __name__ == '__main__':
    main()