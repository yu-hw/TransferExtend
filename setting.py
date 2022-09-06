opt = {}

opt['vocabSize'] = 30000
opt['embedSize'] = 64

opt['encoder'] = {}
opt['encoder']['embedSize'] = opt['embedSize']
opt['encoder']['hiddenSize'] = 64
opt['encoder']['numLayers'] = 2
opt['encoder']['dropout'] = 0.3