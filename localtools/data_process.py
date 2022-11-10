import pickle
import javalang
import os
import re
import string
import random

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.readlines()


def write_file(filepath, str):
    with open(filepath, "w") as f:
        f.write(str)


def solve_camel_and_underline(token):
    if token.isdigit():
        return [token]
    else:
        p = re.compile(r'([a-z]|\d)([A-Z])')
        sub = re.sub(p, r'\1_\2', token).lower()
        sub_tokens = sub.split("_")
        tokens = re.sub(" +", " ", " ".join(sub_tokens)).strip()
        final_token = []
        for factor in tokens.split(" "):
            final_token.append(factor.rstrip(string.digits))
        return final_token


def javalang_tokens_solve(javalang_tokens):
    tokens = []
    for token in javalang_tokens:
        if isinstance(token, javalang.tokenizer.String):
            tmp_token = ["stringliteral"]
        else:
            tmp_token = solve_camel_and_underline(token.value)
        tokens += tmp_token
    return tokens


if __name__ == '__main__':
    print("### 读取数据集")
    dataset_pkl_path = '../data/dataset.pkl'
    dataset = read_pkl(dataset_pkl_path)
    print("### 获取参数")
    max_length = 500
    faultTypes = dataset.keys()
    # faultTypes = ['MutateDataType']
    data = {}

    print("### 处理文件")
    for faultType in faultTypes:
        print(f"[{faultType}]")
        data_length = 0
        # positive: 正确标记
        # patch: positive fix
        # negative: 错误标记
        for type in dataset[faultType].keys():
            if(data_length != 0):
                assert data_length == len(dataset[faultType][type])
            else:
                data_length = len(dataset[faultType][type])
        print(f"{faultType} length: {data_length}")
        
        data_text = {}
        data_text['beforefix'] = []
        data_text['afterfix'] = []
        data_text['label'] = []
        
        for i in range(data_length):    
            data_text['beforefix'].append(dataset[faultType]['positive'][i])
            data_text['afterfix'].append(dataset[faultType]['patch'][i])
            data_text['label'].append(1)
            data_text['beforefix'].append(dataset[faultType]['negative'][i])
            data_text['afterfix'].append(dataset[faultType]['negative'][i].replace("rank2fixstart", " ").replace("rank2fixend", " "))
            data_text['label'].append(0)
        data_text_length = len(data_text['label'])
        
        data_token = {}
        data_token['beforefix'] = []
        data_token['afterfix'] = []
        data_token['label'] = []
        data_token['beforefix'] = []
        data_token['afterfix'] = []
        data_token['label'] = []
        
        for i in range(data_text_length):
            # 将驼峰命名和下划线命名强制分离
            beforefix   =   javalang_tokens_solve(javalang.tokenizer.tokenize(data_text['beforefix'][i]))
            afterfix    =   javalang_tokens_solve(javalang.tokenizer.tokenize(data_text['afterfix'][i]))
            label       =   data_text['label'][i]
            if(len(beforefix) <= max_length and len(afterfix) <= max_length):
                data_token['beforefix'].append(beforefix)
                data_token['afterfix'].append(afterfix)
                data_token['label'].append(label)
        data_token_num = len(data_token['label'])
        
        print(f"处理后数据量为: {data_token_num}")
        
        write_path = os.path.join("../data", faultType)
        os.makedirs(write_path, exist_ok=True)
        write_file(os.path.join(write_path, 'beforefix.txt'), '\n'.join([' '.join(line) for line in data_token['beforefix']]))
        write_file(os.path.join(write_path, 'afterfix.txt'), '\n'.join([' '.join(line) for line in data_token['afterfix']]))
        write_file(os.path.join(write_path, 'label.txt'), '\n'.join([str(label) for label in data_token['label']]))
            
        train_num = int(0.8 * data_token_num)
        valid_num = int(0.1 * data_token_num)
        shuffleSeed = 7890
        
        idx = list(range(data_token_num))
        random.seed(shuffleSeed)
        random.shuffle(idx)
        
        train = {'source': [], 'target': [], 'label': []}
        valid = {'source': [], 'target': [], 'label': []}
        test = {'source': [], 'target': [], 'label': []}
        
        for i, id in enumerate(idx):
            if i < train_num:
                train['source'].append(data_token['beforefix'][id])
                train['target'].append(data_token['afterfix'][id])
                train['label'].append(data_token['label'][id])
            elif i < train_num + valid_num:
                valid['source'].append(data_token['beforefix'][id])
                valid['target'].append(data_token['afterfix'][id])
                valid['label'].append(data_token['label'][id])
            else:
                test['source'].append(data_token['beforefix'][id])
                test['target'].append(data_token['afterfix'][id])
                test['label'].append(data_token['label'][id])
                
        print(f"train num: {len(train['label'])}")
        print(f"valid num: {len(valid['label'])}")
        print(f"test  num: {len(test['label'])}")
        
        write_pkl(os.path.join(write_path, "train.pkl"), train)
        write_pkl(os.path.join(write_path, "valid.pkl"), valid)
        write_pkl(os.path.join(write_path, "test.pkl"), test)
