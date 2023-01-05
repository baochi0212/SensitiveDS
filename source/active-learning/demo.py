import argparse
import numpy as np
import torch
from pprint import pprint
import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime
from transformers import BertConfig
from data_utils import load_data, text2dict
from utils import get_dataset, get_net, get_strategy
from transformers import AutoTokenizer


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
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--decay', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--temp', type=float, default=0.1)
''' Environment '''
parser.add_argument('--backend', default=False, action='store_true')
parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')


parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
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
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

train_dataset, test_dataset = get_dataset(args.dataset_name, args)
train_dataloader, test_dataloader = load_data(dataset=args.dataset,
                                                      data_dir=args.data_dir,
                                                      tokenizer=tokenizer,
                                                      train_batch_size=args.train_batch_size,
                                                      test_batch_size=args.test_batch_size,
                                                      model_name=args.model_name,
                                                      method=args.method,
                                                      workers=0)

# dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device, args)                   # load network

strategy = get_strategy(args.strategy_name)(train_dataset, net)  # load strategy
# # start experiment
# dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
# print(f"number of unlabeled pool: {train_dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {len(test_dataset)}")

# round 0 accuracy
print("Round 0")
# strategy.train()
preds = strategy.predict(test_dataloader)
# print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)

#     # update labels
#     strategy.update(query_idxs)
#     if not args.dataset_name:
#         strategy.train(args)
#     else:
#         strategy.train()


#     # calculate accuracy
#     preds = strategy.predict(dataset.get_test_data())
#     print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
