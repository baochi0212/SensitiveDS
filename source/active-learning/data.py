import numpy as np
import torch
from torchvision import datasets
import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import fuckit
from sklearn.model_selection import train_test_split
data_dir = "./data"
class UnlabeledSet(Dataset):
    def __init__(self, unlabeled_dataset):
        self.unlabeled_dataset = unlabeled_dataset
    def __len__(self):
        return len(self.unlabeled_dataset)
    def __getitem__(self, idx):
        return self.unlabeled_dataset[idx]
class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method, mode='train', args=None):
        #MODE
        self.mode = mode

        #DATASET
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        sep_token = ['[SEP]'] if model_name in ['bert', 'phobert'] else ['</s>']
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label = '[' + str(data['label']) + ']' if '[' in list(label_dict.keys())[0] else str(data['label'])
            if  label not in label_dict:
                label_id = 0
            else:
                label_id = label_dict[label]

            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset
        #ACTIVE LEARNING
        if mode == 'train':
                
            self.n_pool = len(dataset)
            #labeled state
            self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
            #init 100 for round 0:
            self.init_idxs = np.arange(args.n_init_labeled)
            self.labeled_idxs[self.init_idxs] = True
            #labeled set for training
            self.labeled_dataset = []
            self.unlabeled_dataset = []

            for i in range(len(self._dataset)):
                if self.labeled_idxs[i]:
                    self.labeled_dataset.append(self._dataset[i])
                else:
                    self.unlabeled_dataset.append(self._dataset[i])
                    
          

    def __getitem__(self, index):
        if self.mode != 'train':
            return self._dataset[index]
        else:
            return self.labeled_dataset[index]

    def __len__(self):
        if self.mode != 'train':
            return len(self._dataset)
        else:
            return len(self.labeled_dataset)


    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        #in the reverse direction
        return np.where(self.labeled_idxs == False)[0], self.unlabeled_dataset
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        #in the reverse direction
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

def get_Sensitive(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model_name = args.model_name
    method = args.method
    if args.dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
    if args.dataset == 'sensitive':
        train_data = json.load(open(os.path.join(data_dir, 'sensitive_train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'sensitive_test.json'), 'r', encoding='utf-8'))
    label_dict = args.label_dict
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method, mode='train', args=args)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method, mode='test', args=args)
    
    return trainset, testset