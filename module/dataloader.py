import torch
import random

class PreDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        if(len(data['source']) != len(data['target']) or len(data['source']) != len(data['label'])):
            print("the data is not aligned ")
            raise RuntimeError
        
        self.source = data['source']
        self.target = data['target']
        self.label = data['label']
        self.len = len(self.source)
        self.idx = range(self.len)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.source[self.idx[idx]], self.target[self.idx[idx]], self.label[self.idx[idx]]

def build_dataloader(opt, data):
    batch_size = opt['batch_size']
    shuffle = opt['shuffle']
    
    dataset = PreDataSet(data)
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)