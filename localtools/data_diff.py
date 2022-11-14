import pickle
import os

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


if __name__ == '__main__':
    print("### 读取数据集")
    dataset_pkl_path = '/home/LAB/caohl/TransferExtend/data/dataset.pkl'
    dataset = read_pkl(dataset_pkl_path)
    print("### 获取参数")
    faultTypes = dataset.keys()
    # faultTypes = ['MutateDataType']
    data = {}

    print("### 处理文件")
    tmpPath = "/home/LAB/caohl/TransferExtend/data/tmp";
    for faultType in faultTypes:
        print(f"[{faultType}]");
        data[faultType] = {'beforefix': [], 'afterfix': [], 'diff': [], 'label': []}
        beforefixPath = os.path.join(tmpPath, "beforefix.txt")
        afterfixPath = os.path.join(tmpPath, "afterfix.txt")
        exampleNum = len(dataset[faultType]['positive'])
        print(f"num = {exampleNum}")
        for i in range(exampleNum):
            beforefix = dataset[faultType]['positive'][i]
            afterfix  = dataset[faultType]['patch'][i]
            write_file(beforefixPath, beforefix);
            write_file(afterfixPath,  afterfix);
            data[faultType]['beforefix'].append(beforefix)
            data[faultType]['afterfix'].append(afterfix)
            data[faultType]['label'].append(1)
            data[faultType]['diff'].append(os.popen("diff -w " + beforefixPath + " " + afterfixPath).read())
            beforefix = dataset[faultType]['negative'][i]
            afterfix  = dataset[faultType]['negative'][i]
            data[faultType]['beforefix'].append(beforefix)
            data[faultType]['afterfix'].append(afterfix)
            data[faultType]['label'].append(0)
            data[faultType]['diff'].append("nochange")
    
    data_diff_pkl_path = '/home/LAB/caohl/TransferExtend/data/dataset_diff.pkl'
    write_pkl(data_diff_pkl_path, data)