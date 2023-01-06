import numpy as np
from .strategy import Strategy
from data import UnlabeledSet
from demo import collate_fn
from torch.utils import data

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        unlabeled_dataset = UnlabeledSet(unlabeled_data)
        loader = data.DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        probs = self.predict_prob(loader)
        print("MLE", probs)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

