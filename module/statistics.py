import time
import math

class Statistics(object):
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.start_time = time.time()
        
    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        
    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)
    
    def xent(self):
        return self.loss / self.n_words
    
    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))
    
    def elapsed_time(self):
        return time.time() - self.start_time