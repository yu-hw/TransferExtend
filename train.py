import model
from setting import * 

def buildEncoder(opt):
    vocabSize = opt['vocabSize']
    embedSize = opt['encoder']['embedSize']
    hiddenSize = opt['encoder']['hiddenSize']
    numLayers = opt['encoder']['numLayers']
    dropout = opt['encoder']['dropout']
    
    return model.Encoder(vocabSize, embedSize, hiddenSize, numLayers, dropout)


def train(opt, net, iterator, optimizer, trainLoss):
    device = opt['device']
    
    net.to(device)
    net.train()