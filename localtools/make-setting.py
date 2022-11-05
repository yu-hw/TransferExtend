from distutils.errors import PreprocessError
import json
import os


def origin_opt():
    opt = {}
    opt['embed_size'] = 64
    opt['hidden_size'] = 100
    opt['lstm_dropout'] = 0
    opt['mlp_dropout'] = 0
    opt['learning_rate'] = 0.001 # Adam
    opt['epoch'] = 40
    opt['NMT_weight'] = 0
    opt['MLP_weight'] = 1
    opt['max_pred_len'] = 300
    opt['batch_size'] = 64
    opt['fault_type'] = "MutateDataType"
    opt['mlp_layer_size'] = [opt['hidden_size'], 2]
    return opt

def process_opt(opt):
    opt['data'] = {}
    opt['data']['fault_type'] = opt['fault_type']
    opt['data']['shuffle'] = True
    opt['data']['batch_size'] = opt['batch_size']

    opt['vocab'] = {}
    opt['vocab']['vocab_size'] = 30000
    opt['vocab']['share_vocab'] = True

    opt['encoder'] = {}
    opt['encoder']['vocab_size'] = opt['vocab']['vocab_size']
    opt['encoder']['embed_size'] = opt['embed_size']
    opt['encoder']['hidden_size'] = opt['hidden_size'] // 2
    opt['encoder']['num_layers'] = 2
    opt['encoder']['dropout'] = opt['lstm_dropout']

    opt['decoder'] = {}
    opt['decoder']['vocab_size'] = opt['vocab']['vocab_size']
    opt['decoder']['embed_size'] = opt['embed_size']
    opt['decoder']['hidden_size'] = opt['hidden_size']
    opt['decoder']['num_layers'] = 2
    opt['decoder']['dropout'] = opt['lstm_dropout']

    opt['mlp'] = {}
    opt['mlp']['layer_size'] = opt['mlp_layer_size']
    opt['mlp']['dropout'] = opt['mlp_dropout']

    
def main():
    fault_type = []
    data_path = '/home/LAB/caohl/TransferExtend/data'
    for root, dirs, files in os.walk(data_path):
        if 'train.pkl' in files:
            name = os.path.split(root)[-1]
            fault_type.append(name)
    
    os.mkdir
    json_path = '/home/LAB/caohl/TransferExtend/train-setting'
    for name in fault_type:
        opt = origin_opt()
        opt['fault_type'] = name
        process_opt(opt)
        with open(os.path.join(json_path, name + '.json'), 'w') as f:
            f.write(json.dumps(opt))
    

if __name__ == '__main__':
    main()