import numpy as np
from .strategy import Strategy
from data import UnlabeledSet
import torch
from functools import partial
from torch.utils import data
def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)
class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n, collate_fn=None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        print("---------UNLABELED DATA CHECK------------", unlabeled_idxs.shape, len(unlabeled_data))
        unlabeled_dataset = UnlabeledSet(unlabeled_data)
        loader = data.DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        probs = self.predict_prob(loader)
        uncertainties = torch.max(probs, dim=1)[0]
        #check the confusion
        print(f"MIN PROB vs MAX PROB {uncertainties.sort()[0][:n].numpy()[0], uncertainties.sort()[0][:n].numpy()[-1]}")
        print(f"MIN vs MAX UNSELECTED{uncertainties.sort()[0].numpy()[0], uncertainties.sort()[0].numpy()[-1]}")
        print("TESTING INDEXES", unlabeled_idxs[uncertainties.sort()[1][:n].numpy()])
        return unlabeled_idxs[uncertainties.sort()[1][:n].numpy()]
