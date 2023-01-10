import argparse
import numpy as np
import torch
from pprint import pprint
import os
import sys
import time
import torch
from functools import partial
import random
import logging
import argparse
from datetime import datetime
from transformers import BertConfig
from data_utils import load_data, text2dict
from utils import get_dataset, get_net, get_strategy
from transformers import AutoTokenizer
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


parser = argparse.ArgumentParser()
num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2, 'phoATIS': 25, 'uit-nlp': 3, 'sensitive': 5}
''' Base '''
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset', type=str, default='sst2', choices=num_classes.keys())
parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'roberta', 'phobert', 'bert-scratch'])
parser.add_argument('--method', type=str, default='dualcl', choices=['ce', 'scl', 'dualcl'])
''' Optimization '''
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--temp', type=float, default=0.1)
''' Environment '''
parser.add_argument('--backend', default=False, action='store_true')
parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--pretrained_model', type=str, default='bert-base-cased')

parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=100, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default=None, choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "sensitive"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
args.num_classes = num_classes[args.dataset]
args.device = torch.device(args.device)
args.log_name = '{}_{}_{}_{}.log'.format(args.dataset, args.model_name, args.method, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
if not os.path.exists('logs'):
    os.mkdir('logs')
pprint(vars(args))
#LABEL DICT
if args.dataset == 'sst2':
    args.label_dict = {'positive': 0, 'negative': 1}
elif args.dataset == 'sensitive':
    args.label_dict = {'insult': 0, 'religion': 1, 'terrorism': 2, 'politics': 3, 'neutral': 4}



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False





# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
train_dataset, test_dataset = get_dataset(args.dataset_name, args)
# train_dataloader, test_dataloader = load_data(dataset=args.dataset,
#                                                       data_dir=args.data_dir,
#                                                       tokenizer=tokenizer,
#                                                       train_batch_size=args.train_batch_size,
#                                                       test_batch_size=args.test_batch_size,
#                                                       model_name=args.model_name,
#                                                       method=args.method,
#                                                       workers=0)
# load dataset
collate_fn = partial(my_collate, tokenizer=tokenizer, method=args.method, num_classes=len(args.label_dict))
train_dataloader = data.DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
test_dataloader = data.DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)

net = get_net(args.dataset_name, device, args)                   # load network
'''
** interation:
    - round 0: 
        - net.test(TEST_SET)
        - check the number of labeled/unlabeled/testing samples
    - for i in 1:n:
        round i:
            - idxs = strategy.query()
            - TRAIN_SET.update(idxs)
            - NET.train(TRAIN_SET)
            - NET.test(TEST_SET)
'''
#STRATEGY OBJECT    
strategy = get_strategy(args.strategy_name)(train_dataset, net)  # load strategy
# # start experiment
# dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {len(train_dataset._dataset)}")
print(f"number of testing pool: {len(test_dataset)}")

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(test_dataloader)

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")
    #RELOAD MODEL
    print(f"..........RELOAD MODEL...........")
    strategy.net = get_net(args.dataset_name, device, args) 
    #QUERY SAMPLES
    query_idxs = strategy.query(args.n_query, collate_fn=collate_fn)

    #UPDATE TRAIN_SET
    strategy.update(query_idxs)
    #LOADER
    train_dataloader = data.DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = data.DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    print("------------------------------NUM LABELED", sum(train_dataset.labeled_idxs), len(train_dataset.labeled_dataset))
    print("------------------------------NUM BATCH", len(train_dataloader))
    #TRAIN
    if not args.dataset_name:
        strategy.train(args, train_dataloader, test_dataloader)
    else:
        strategy.train(args, train_dataloader, test_dataloader)


    # #PREDICTION 
    # preds = strategy.predict(test_dataloader)
