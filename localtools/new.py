import pickle

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

path = "/home/LAB/yuhw/SE/data/dataset_pre_diff_11.10.pkl"

data = read_pkl(path)
print(data.keys())
