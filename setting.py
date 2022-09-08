opt = {}

opt['embed_size'] = 64
opt['hidden_size'] = 128
opt['dropout'] = 0.3

opt['vocab'] = {}
opt['vocab']['vocab_size'] = 30000
opt['vocab']['share_vocab'] = True

opt['encoder'] = {}
opt['encoder']['embed_size'] = opt['embed_size']
opt['encoder']['hidden_size'] = opt['hidden_size'] // 2
opt['encoder']['num_layers'] = 2
opt['encoder']['dropout'] = opt['dropout']

opt['decoder'] = {}
opt['decoder']['embed_size'] = opt['embed_size']
opt['decoder']['hidden_size'] = opt['hidden_size']
opt['decoder']['num_layers'] = 2
opt['decoder']['dropout'] = opt['dropout']