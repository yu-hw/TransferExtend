

def add_bos_eos(opt, data):
    bos = opt['vocab']['tgt_bos']
    eos = opt['vocab']['tgt_eos']
    workType = ['train', 'valid', 'test']
    for type0 in workType:
        data[type0]['target'] = [[bos] + line + [eos] for line in data[type0]['target']]


def truncate_pad(data, num_steps, padding_token):
    if len(data) > num_steps:
        return data[:num_steps]  # 截断
    return data + [padding_token] * (num_steps - len(data))  # 填充


def align_data(opt, data):
    padding_idx = {'source': opt['vocab']['src_pad'], 'target': opt['vocab']['tgt_pad']}
    workType = ['train', 'valid', 'test']
    dataType = ['source', 'target']
    
    for type0 in workType:
        for type1 in dataType:
            data[type0][type1 + "_length"] = []
            for line in data[type0][type1]:
                data[type0][type1 + "_length"].append(len(line))
    
    for type0 in workType:
        ll = max(data[type0][type1 + "_length"])
        for type1 in dataType:
            data[type0][type1] = [truncate_pad(
                line, ll, padding_idx[type1]) for line in data[type0][type1]]