import pickle

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_data(opt):
    fault_type = opt['data']['fault_type']
    root_path = '/home/LAB/caohl/TransferExtend/data/' + fault_type

    dataType = ['train', 'valid', 'test']

    data = {}
    for type in dataType:
        data[type] = read_pkl(root_path + '/' + type + '.pkl')
    return data