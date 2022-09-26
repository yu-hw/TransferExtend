import torch
import random


class PreDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.source = torch.tensor(data['source'])
        self.target = torch.tensor(data['target'])
        self.label = torch.tensor(data['label'])
        self.source_length = torch.tensor(data['source_length'])
        self.target_length = torch.tensor(data['target_length'])
        self.len = len(self.source)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx], self.label[idx], self.source_length[idx], self.target_length[idx]


def build_dataloader(opt, data):
    batch_size = opt['data']['batch_size']
    shuffle = opt['data']['shuffle']

    dataset = PreDataSet(data)
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
