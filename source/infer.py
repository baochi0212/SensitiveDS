import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
import torch.nn as nn
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data, text2dict
from transformers import logging, AutoTokenizer, AutoModel, BertModel, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
label_dict = {'insult': 0, 'religion': 1, 'terrorism': 2, 'politics': 3, 'neutral': 4}
label_index = dict([(value, key) for key, value in label_dict.items()])
save_path = os.environ['SAVE_MODEL']
device = 'cuda'
args, logger = get_config()
input = "The Justice Department accused Naser Almadaoji, an Iraqi-born U.S. citizen, of arranging with an Islamic State supporter to travel to Afghanistan, where he intended to train with a group called ISIS Wilayat Khorasan. The supporter was actually an FBI informant, and federal agents arrested Almadaoji at the Columbus airport before he departed in October 2018."
input = input.split('.')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
input = tokenizer(input, return_tensors='pt')
input = dict([(key, value.to(device)) for key, value in input.items()])
base_model = AutoModel.from_pretrained('bert-base-uncased')
model = Transformer(base_model, args.num_classes, args.method)
model.load_state_dict(torch.load(save_path + '/best_model.mdl'))
model = model.to(device)
output = torch.argmax(model(input)['predicts']).item()
#output
print("OUTPUT", [label_index[label.item() for label in output]])