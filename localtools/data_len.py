import os
import pickle
import matplotlib.pyplot as plt
import numpy

def read_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_pkl(filepath, obj):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read()


def write_file(filepath, str):
    with open(filepath, "w") as f:
        f.write(str)
        

def work(l):
    num = len(l)
    l.sort()
    pers = list(range(5, 100, 5))
    for per in pers:
        print(f"{per}: {l[int(num * per / 100)]}")
    print(f"max len = {l[-1]}")


fault_types = []
data_path = '/home/LAB/caohl/TransferExtend/data/data-diff'
for root, dirs, files in os.walk(data_path):
    if 'beforefix_length.txt' in files:
        name = os.path.split(root)[-1]
        print(f"[{name}]")
        before_l = [int(x) for x in read_file(os.path.join(root, 'beforefix_length.txt')).split('\n')]
        after_l = [int(x) for x in read_file(os.path.join(root, 'afterfix_length.txt')).split('\n')]
        print("before fix")
        work(before_l)
        print("after  fix")
        work(after_l)