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

save_path = os.environ['SAVE_MODEL']
device = 'cuda'
args, logger = get_config()
input = 'CHi dep trai is going to fuck with you'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
base_model = AutoModel.from_pretrained('bert-base-uncased')
model = Transformer(base_model, args.num_classes, args.method)
model = model.load_state_dict(torch.load(save_path))
model.to(device)
print(model(tokenizer(input)))