import torch
import wandb
from sklearn.metrics import accuracy_score
import time
import torch.optim as optim
from modules.model import IntentClassifier
import torch.nn as nn
import numpy as np


class Trainer():
    
    def __init__(self, model, device, metric_fn, optimizer=None, scheduler=None, logger=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.metric_fn = metric_fn
        
    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        target_lst = []
        pred_lst = []
        for batch_index, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            src = data[:, 0, :]
            segs = data[:, 1, :]
            mask = data[:, 2, :]
            output = self.model(src, mask, segs, target)
            loss = output.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.train_total_loss += loss
            target_lst.extend(target.cpu().tolist())
            pred = output.logits.argmax(dim=1)
            pred_lst.extend(pred.cpu().tolist())
            
            batch_score = self.metric_fn(target_lst, pred_lst)
            
            msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader)} mean loss: {loss} score: {batch_score}"
            if batch_index%100 == 0:
                if self.logger:
                    self.logger.info(msg)
                print(msg)
            
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = accuracy_score(y_true=target_lst, y_pred=pred_lst)
        msg = f'Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)
        
        return self.train_mean_loss, self.train_score
        
    def validate_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        target_lst = []
        pred_lst = []
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                target = target.to(self.device)
                src = data[:, 0, :]
                segs = data[:, 1, :]
                mask = data[:, 2, :]
                output = self.model(src, mask, segs, target)
                loss = output.loss
                self.val_total_loss += loss
                target_lst.extend(target.tolist())
                pred_lst.extend(output.logits.argmax(dim=1).tolist())
            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = accuracy_score(y_true=target_lst, y_pred=pred_lst)
            msg = f'Epoch {epoch_index}, Validation, loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)
            
        return self.val_mean_loss, self.validation_score

    def test_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        target_lst = []
        pred_lst = []
        logits = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for batch_index, (data) in enumerate(dataloader):
                data = data.to(self.device)
                src = data[:, 0, :]
                segs = data[:, 1, :]
                mask = data[:, 2, :]
                output = self.model(src, mask, segs)
                logits = torch.cat([logits, output.logits], dim=0)
                pred_lst.extend(output.logits.argmax(dim=1).tolist())
                
                if batch_index % 100 == 0:
                    print(f'Prediction: {batch_index} batch completed')
        
        return pred_lst, logits
