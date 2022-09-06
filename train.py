import model
from setting import * 

def buildEncoder(opt):
    vocabSize = opt['vocabSize']
    embedSize = opt['encoder']['embedSize']
    hiddenSize = opt['encoder']['hiddenSize']
    numLayers = opt['encoder']['numLayers']
    dropout = opt['encoder']['dropout']
    
    return model.Encoder(vocabSize, embedSize, hiddenSize, numLayers, dropout)


list = [[1, 2, 3, 4], [5, 6, 7, 8]]
encoder = buildEncoder(opt)
memory_bank, lengths, hidden_state = encoder(list)
print("1")