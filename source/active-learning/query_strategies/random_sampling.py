import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n, collate_fn=None):
        idxs = np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
        print("idxs", idxs)
        return idxs
