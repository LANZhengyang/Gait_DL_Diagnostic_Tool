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


class Inception_module(pl.LightningModule):
    def __init__(self, input_c, kernel_size=41, bottleneck_size=32, nb_filters = 32):
        super().__init__()
        
        self.kernel_size= kernel_size - 1
        self.bottleneck_size = bottleneck_size
        self.nb_filters = nb_filters
        
        self.conv1 = nn.Conv1d(input_c, self.bottleneck_size, 1, bias=False)
        
        self.kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        
        self.conv_list =  nn.ModuleList([nn.Conv1d(self.bottleneck_size, self.bottleneck_size, i, bias=False) for i in self.kernel_size_s])            

        self.maxpool1d = nn.MaxPool1d(3, stride=1)
        
        self.conv2  = nn.Conv1d(input_c, self.bottleneck_size, 1, bias=False)
        
        self.BN = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        x_conv1 = self.conv1(x)
        
        x_list = []
        for i, net in enumerate(self.conv_list):
            
            pad_size = self.kernel_size_s[i]//2
            if self.kernel_size_s[i]%2 == 0:
                pad_r = pad_size - 1
                pad_l = pad_size
            else:
                pad_r = pad_size
                pad_l = pad_size
            
            x_t = F.pad(x_conv1, (pad_r,pad_l,0,0)) # [left, right, top, bot]
            x_list.append(net(x_t))
    
        x_t = F.pad(x, (1,1,0,0))
        x_t = self.maxpool1d(x_t)
        

        x_t = self.conv2(x_t)
        
        x_list.append(x_t)
        
        x = torch.cat(x_list, 1)
        
        x = self.BN(x)
        
        x = self.relu(x)
        
        return x
         
class Shortcut_layer(pl.LightningModule):
    def __init__(self,input_c,output_c):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_c , output_c, kernel_size=1, bias=False)
        self.BN = nn.BatchNorm1d(output_c)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        input_tensor, output_tensor = x
        
        short_cut_y = self.conv1(input_tensor)
        short_cut_y = self.BN(short_cut_y)
        
        x = torch.add(short_cut_y,output_tensor)
        x = self.relu(x)
        
        return x
class Inception(pl.LightningModule):
    def __init__(self, nb_classes=2, lr=0.001, lr_factor=0.5, lr_patience=50, use_residual=True, depth=6, lr_reduce = True):
        super().__init__()
        
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_reduce = lr_reduce
        
        self.use_residual = use_residual
        self.depth = depth
        
        self.model_inception_list = nn.ModuleList([])
        self.model_shortcut_list = nn.ModuleList([])
        
        model_inception_list_c = [22,128,128,128,128,128]
        model_shortcut_in_list_c = [22,128]
        model_shortcut_out_list_c = [128,128]
    
        for i in range(self.depth):
            self.model_inception_list.append(Inception_module(model_inception_list_c[i]))
            
            if self.use_residual and i % 3 == 2:
                
                self.model_shortcut_list.append( Shortcut_layer(model_shortcut_in_list_c[int((i-2)/3)],model_shortcut_out_list_c[int((i-2)/3)]))
        
        self.globalavgpooling1d = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(4*32 , nb_classes )
        
        self.softmax = torch.nn.Softmax()
  
    def forward(self,x):
        
        x_res_input = x
        
        for i in range(self.depth):

            x = self.model_inception_list[i](x)
            if self.use_residual and i % 3 == 2:
                x = self.model_shortcut_list[int((i-2)/3)]([x_res_input,x])
                x_res_input = x
        
        x = self.globalavgpooling1d(x)
        x = torch.squeeze(x)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    def training_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())

        if len(y_true)==1:
            y_pre = torch.unsqueeze(y_pre, 0)
        
        loss = nn.NLLLoss()(torch.log(y_pre),y_true.type(torch.cuda.LongTensor))
        
        accuracy = torch.tensor([torch.argmax(i) for i in y_pre])
        accuracy = FM.Accuracy().cuda()(accuracy.int().cuda(), y_true.int().cuda())

        self.log("train_accuracy",accuracy)
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch

        y_pre = self.forward(x.float())
        
        if len(y_true)==1:
            y_pre = torch.unsqueeze(y_pre, 0)
    
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
        
    
class Classifier_InceptionTime(DL_classifier):
    def __init__(self, nb_classes, lr =0.001, lr_factor = 0.5, lr_patience=50, lr_reduce = True, batch_size=64, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir=None):
        self.model = Inception(nb_classes, lr, lr_factor, lr_patience,lr_reduce = lr_reduce)
        super().__init__(self.model, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        