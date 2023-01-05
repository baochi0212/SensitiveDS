import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data, text2dict
from transformers import logging, AutoTokenizer, AutoModel, BertModel, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf = self.net().to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
        

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
class Netformer(nn.Module):
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model 
        self.args = args

    def predict_prob(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        predicts = F.softmax(predicts, -1)
        return predicts.cpu()
    def train(self, args):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.base_model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr)
        total_steps = len(train_dataloader)*args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=total_steps, num_warmup_steps=50)
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer, scheduler)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
    def predict(self, dataloader):
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        test_loss, n_correct, n_test = 0, 0, 0
        y_true, y_pred = [], []
        self.base_model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.base_model(inputs)
                print("OUTPUTS", outputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                print(targets) #for verify sanity
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                y_pred += torch.argmax(outputs['predicts'], -1).cpu().numpy().reshape(-1).tolist()
                y_true += targets.cpu().numpy().reshape(-1).tolist()
                n_test += targets.size(0)
        print(classification_report(y_true, y_pred))
        return test_loss / n_test, n_correct / n_test

    


class Transformer(nn.Module):

    def __init__(self, base_model, num_classes, method, args):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.args = args
        for param in base_model.parameters():
            param.requires_grad_(True)

        
        

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs


    
    


