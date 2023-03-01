import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import copy


class Dataset_torch(Dataset):

    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]
    

class DL_classifier:
        def __init__(self,model, batch_size, earlystopping, et_patience, max_epochs, gpu, default_root_dir,nb_clases=2):
            self.model = model
            self.batch_size = batch_size
            self.earlystopping = earlystopping
            self.et_patience = et_patience
            self.max_epochs = max_epochs
            self.gpu = gpu
            self.default_root_dir = default_root_dir
            
        def load(self,model_dir,nb_classes=2):
            self.model = self.model.load_from_checkpoint(model_dir,nb_classes = nb_classes)
            
            return self
            
        def fit(self, x_train, y_train, x_val, y_val,default_root_dir=None,ckpt_monitor=None):
            
#           
            if default_root_dir != None:
                self.default_root_dir = default_root_dir
            
            checkpoint_callback = ModelCheckpoint(
                monitor=ckpt_monitor,
                dirpath=self.default_root_dir,
                filename="model-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}",
                save_top_k=1,
                mode="max",
            )
            
            train_set = Dataset_torch([x_train, y_train])
            data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size,shuffle=True,num_workers=0)
            test_set = Dataset_torch([x_val, y_val])
            data_loader_test = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size,shuffle=True,num_workers=0)
            
            tb_logger = pl_loggers.TensorBoardLogger(self.default_root_dir+"/logs/")
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            
            if self.earlystopping:           
                early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=self.et_patience,
                verbose=True)

                self.trainer = pl.Trainer(gpus=self.gpu,max_epochs= self.max_epochs, logger=tb_logger, callbacks=[early_stop_callback,checkpoint_callback,lr_monitor])
                self.trainer.fit(self.model, data_loader_train, data_loader_test)

            if not self.earlystopping:
                self.trainer = pl.Trainer( gpus=self.gpu,max_epochs= self.max_epochs,logger=tb_logger, callbacks=[checkpoint_callback,lr_monitor])
                self.trainer.fit(self.model, data_loader_train, data_loader_test)
            return self

        def predict(self, x_pred, batch_size=64, gpu=[0]):
            pred_set = Dataset_torch(x_pred,with_label=False)
            data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=batch_size,num_workers=4)
            if not torch.cuda.is_available():
                gpu = None
            trainer = pl.Trainer(gpus=gpu)
            pred = trainer.predict(model=self.model,dataloaders = data_loader_pred)

            y_pre = torch.tensor([torch.argmax(i) for i in torch.cat(pred)])

            return y_pre
        
          
            
    