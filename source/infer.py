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

def get_prediction(input):
    text_input = [text for text in input.split('.') if len(text) > 1]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    input  = tokenizer(text_input,
                            truncation=True,
                            max_length=256,
                            pad_to_max_length=True,
                            return_tensors='pt')
    input = dict([(key, value.to(device)) for key, value in input.items()])
    base_model = AutoModel.from_pretrained('bert-base-uncased')
    model = Transformer(base_model, args.num_classes, args.method)
    model.load_state_dict(torch.load(save_path + '/best_model.mdl'))
    model.eval()
    model = model.to(device)
    output = torch.argmax(model(input)['predicts'], dim=-1)
    return [(text_input[i], label_index[output[i].item()], torch.max(torch.nn.functional.softmax(model(input)['predicts'][i], -1), -1)[0].item()) for i in range(len(text_input))]

if __name__ == '__main__':
    input = "Fuck you bastard"
    print(label_index)
    print("PREDICTION", get_prediction(input))

    #Former President Donald Trump Thursday refused to commit to supporting the 2024 Republican presidential nominee — if he doesn’t win it. “It would depend. I would give you the same answer I gave in 2016,” Trump said in an interview with conservative radio host Hugh Hewitt. “It would have to depend on who the nominee was
    #I said hello. But He call me a fucking bastard


