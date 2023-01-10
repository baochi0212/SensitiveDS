import numpy as np
import torch
from .strategy import Strategy
from torch.utils import data
from data import UnlabeledSet

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n, collate_fn=None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        print("---------UNLABELED DATA CHECK------------", unlabeled_idxs.shape, len(unlabeled_data))
        unlabeled_dataset = UnlabeledSet(unlabeled_data)
        loader = data.DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        probs = self.predict_prob(loader)
        log_probs = torch.log(probs)
        uncertainties = torch.sum(probs*log_probs, dim=1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
