import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics as FM
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from classifiers.DL_classifier import DL_classifier

class LSTM(pl.LightningModule):
    def __init__(self,nb_classes=2, lr =0.001, lr_factor = 0.5, lr_patience=50, lr_reduce = True):
        super().__init__()
        
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_reduce = lr_reduce
        
        self.hidden_neurons = 400
        
        
        self.LSTM_1 = torch.nn.LSTM(input_size=22, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        self.LSTM_2 = torch.nn.LSTM(input_size=800, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        self.LSTM_3 = torch.nn.LSTM(input_size=800, hidden_size= self.hidden_neurons,bidirectional=True,batch_first=True)
        
        self.fc = nn.Linear(self.hidden_neurons * 2,nb_classes)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = torch.nn.Softmax()
        

    def forward(self,x):
        
        x = torch.transpose( x, 1, 2)        
       
        x, _ = self.LSTM_1(x)
        
        x = self.dropout(x)
        
        x, _ = self.LSTM_2(x)
        x = self.dropout(x)
        
        _, (x,_) = self.LSTM_3(x)
        
        x = torch.cat((x[-2,:,:], x[-1,:,:]), dim = 1)
        
        x = self.dropout(x)
        
        x = self.fc(x)  
        
        x = self.softmax(x)
        
        return x
         
    
    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())
     
        loss = nn.NLLLoss()(torch.log(y_pre),y_true.type(torch.cuda.LongTensor))
        
        accuracy = torch.tensor([torch.argmax(i) for i in y_pre])
        accuracy = FM.Accuracy().cuda()(accuracy.int().cuda(), y_true.int().cuda())

        self.log("train_accuracy",accuracy)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())

        loss = nn.NLLLoss()(torch.log(y_pre),y_true.type(torch.cuda.LongTensor))
        
        accuracy = torch.tensor([torch.argmax(i) for i in y_pre])
        
        accuracy = FM.Accuracy().cuda()(accuracy.int().cuda(), y_true.int().cuda())
        
        self.log("val_accuracy",accuracy)
        self.log('val_loss', loss)

        return loss

    def predict_step(self, batch, batch_idx):
        
        x = batch

        y_pre = self.forward(x.float())
        
        if y_pre.shape==torch.Size([2]):
            y_pre = torch.unsqueeze(y_pre, 0)
        
        return y_pre
    
    def configure_optimizers(self):
        
        if self.lr_reduce:

            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,eps=1e-07)
            scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience= self.lr_patience)

            return { 'optimizer': optimizer, 'lr_scheduler': { 'scheduler': scheduler, 'monitor': 'val_loss'} }
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr,eps=1e-07)

    
class Classifier_LSTM(DL_classifier):
    def __init__(self, nb_classes, lr =0.001, lr_factor = 0.5, lr_patience=50, lr_reduce = True, batch_size=64, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir=None):
        self.model = LSTM(nb_classes, lr, lr_factor, lr_patience,lr_reduce = lr_reduce)
        super().__init__(self.model, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)

        