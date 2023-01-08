import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n, collate_fn=None):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        count = 0
        self.dataset.labeled_idxs[pos_idxs] = True
        for i in range(len(self.dataset._dataset)):
                if self.dataset.labeled_idxs[i] and self.dataset._dataset[i] not in self.dataset.labeled_dataset:
                    count += 1
                    self.dataset.labeled_dataset.append(self.dataset._dataset[i])
                if self.dataset.labeled_idxs[i] and self.dataset._dataset[i] in self.dataset.unlabeled_dataset:
                    self.dataset.unlabeled_dataset.remove(self.dataset._dataset[i])
        print("DEBUG COUNT", count) 
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, args=None, train_dataloader=None, test_dataloader=None):
        if args:
            self.net.train(train_dataloader, test_dataloader)
            return


    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

