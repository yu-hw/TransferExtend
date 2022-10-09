from distutils.errors import PreprocessError
import json
import os


def origin_opt():
    opt = {}
    opt['embed_size'] = 128
    opt['hidden_size'] = 256
    opt['dropout'] = 0.3
    opt['learning_rate'] = 0.001 # Adam
    opt['epoch'] = 20
    opt['NMT_weight'] = 0.5
    opt['MLP_weight'] = 0.5
    opt['max_pred_len'] = 300
    opt['batch_size'] = 16
    opt['fault_type'] = "MutateDataType"
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

    
def main():
    # fault_type = []
    # data_path = '/home/LAB/caohl/TransferExtend/data'
    # for root, dirs, files in os.walk(data_path):
    #     if 'train.pkl' in files:
    #         name = os.path.split(root)[-1]
    #         fault_type.append(name)
    
    # json_path = '/home/LAB/caohl/TransferExtend/train-setting'
    # for name in fault_type:
    #     opt = origin_opt()
    #     opt['fault_type'] = name
    #     process_opt(opt)
    #     with open(os.path.join(json_path, name + '.json'), 'w') as f:
    #         f.write(json.dumps(opt))
    opt = origin_opt()
    opt['batch_size'] = 100
    opt['fault_type'] = 'Insert___....'
    process_opt(opt)
    with open('json', 'w') as f:
        f.write(json.dumps(opt))
    
    
    

if __name__ == '__main__':
    main()