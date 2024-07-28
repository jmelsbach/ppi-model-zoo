from pathlib import Path
import time
import json
import os
import sys
from typing import Optional

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, precision_recall_curve

import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import seed as pl_seed

from ranger21 import Ranger21
from dictlogger import DictLogger

from weightdrop import WeightDrop
from nl import Mish
from data import RapppidDataModule, RapppidDataModule2

from passlib import pwd

from tqdm import tqdm
import fire


class MeanClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MeanClassHead, self).__init__()

        if num_layers == 1:
            self.fc = WeightDrop(nn.Linear(embedding_size, 1), ['weight'],
                                    dropout=weight_drop, variational=variational)
        elif num_layers == 2:
            self.fc = nn.Sequential(
                        nn.Linear(embedding_size, embedding_size//2),
                        Mish(),
                        nn.Linear(embedding_size//2, 1))
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        
        z = (z_a + z_b)/2
        z = self.fc(z)

        return z


class MultClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MultClassHead, self).__init__()

        if num_layers == 1:
            self.fc = WeightDrop(nn.Linear(embedding_size, 1), ['weight'],
                                    dropout=weight_drop, variational=variational)
        elif num_layers == 2:
            self.fc = nn.Sequential(
                        WeightDrop(nn.Linear(embedding_size, embedding_size//2), 
                                    ['weight'], dropout=weight_drop, 
                                    variational=variational),
                        Mish(),
                        WeightDrop(nn.Linear(embedding_size//2, 1), 
                                    ['weight'], dropout=weight_drop, 
                                    variational=variational)
                                    )
        else:
            raise NotImplementedError

        self.nl = Mish()

    def forward(self, z_a, z_b):

        z_a = (z_a - z_a.mean()) / z_a.std()
        z_b = (z_b - z_b.mean()) / z_b.std()
        
        z = z_a * z_b

        z = self.nl(z)
        z = self.fc(z)

        return z


class ConcatClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(ConcatClassHead, self).__init__()

        if num_layers == 1:
            self.fc = nn.Linear(embedding_size*2, 1)
        elif num_layers == 2:
            self.fc = nn.Sequential(
                        nn.Linear(embedding_size*2, embedding_size//2),
                        nn.Dropout(weight_drop),
                        Mish(),
                        nn.Linear(embedding_size//2, 1))
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        
        z_ab = torch.cat((z_a, z_b), axis=1)
        z = self.fc(z_ab)

        return z


class ManhattanClassHead(nn.Module):
    def __init__(self):
        super(ManhattanClassHead, self).__init__()

        self.fc = nn.Linear(1, 1)

    def forward(self, z_a, z_b):
        
        distance = torch.sum(torch.abs(z_a-z_b), dim=1).unsqueeze(1)
        y_logit = self.fc(distance)

        return y_logit


class LSTMAWD(pl.LightningModule):
    def __init__(self, num_codes, embedding_size, steps_per_epoch, num_epochs, 
                    lstm_dropout_rate, classhead_dropout_rate, rnn_num_layers, 
                    classhead_num_layers, lr,  weight_decay, bi_reduce, 
                    class_head_name, variational_dropout, lr_scaling, trunc_len, 
                    embedding_droprate, frozen_epochs, optimizer_type):

        super(LSTMAWD, self).__init__()

        if lr_scaling:
            #IMPORTANT: Manual optimization
            self.automatic_optimization = False

        self.lr_scaling = lr_scaling
        self.trunc_len = trunc_len
        self.num_epochs = num_epochs
        self.num_codes = num_codes
        self.steps_per_epoch = steps_per_epoch
        self.embedding_size = embedding_size
        self.lstm_dropout_rate = lstm_dropout_rate
        self.classhead_dropout_rate = classhead_dropout_rate
        self.rnn_num_layers = rnn_num_layers
        self.classhead_num_layers = classhead_num_layers
        self.lr = lr
        self.embedding_droprate = embedding_droprate
        self.weight_decay = weight_decay
        self.bi_reduce = bi_reduce
        self.class_head_name = class_head_name
        self.lr_base = lr
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.nl = Mish()
        self.frozen_epochs = frozen_epochs
        self.optimizer_type = optimizer_type

        if self.bi_reduce == 'concat':
            self.rnn = nn.LSTM(embedding_size, embedding_size//2, rnn_num_layers, bidirectional=True, batch_first=True)
        elif self.bi_reduce in ['max', 'mean', 'last']:
            self.rnn = nn.LSTM(embedding_size, embedding_size, rnn_num_layers, bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Unexpected value for `bi_reduce` {bi_reduce}")

        self.rnn_dp =  WeightDrop(
                self.rnn, ['weight_hh_l0'], lstm_dropout_rate, variational_dropout
            )

        if class_head_name == 'concat':
            self.class_head = ConcatClassHead(embedding_size, 
                                                classhead_num_layers,
                                                classhead_dropout_rate,
                                                variational_dropout)
        elif class_head_name == 'mean':
            self.class_head = MeanClassHead(embedding_size, 
                                                classhead_num_layers,
                                                classhead_dropout_rate,
                                                variational_dropout)
        elif class_head_name == 'mult':
            self.class_head = MultClassHead(embedding_size, 
                                                classhead_num_layers,
                                                classhead_dropout_rate,
                                                variational_dropout)
        elif self.class_head_name == 'manhattan':
            self.class_head = ManhattanClassHead()
        else:
            raise ValueError(f"Unexpected value for `class_head_name` {class_head_name}")

        self.criterion = nn.BCEWithLogitsLoss()
        self.embedding = nn.Embedding(self.num_codes, self.embedding_size, padding_idx=0)

        self.save_hyperparameters()

    def embedding_dropout(self, embed, words, p=0.2):
        """
        Taken from original authors code.
        TODO: re-write and add test
        """
        if not self.training:
            masked_embed_weight = embed.weight
        elif not p:
            masked_embed_weight = embed.weight
        else:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
            masked_embed_weight = mask * embed.weight
    
        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
    
        X = F.embedding(words, masked_embed_weight,
                        padding_idx, embed.max_norm, embed.norm_type,
                        embed.scale_grad_by_freq, embed.sparse)
        return X

    def forward(self, x):
        # Truncate to longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:,:max_len]

        x = self.embedding_dropout(self.embedding, x, p=self.embedding_droprate)
        output, (hn, cn) = self.rnn_dp(x)

        if self.bi_reduce == 'concat':
            # Concat both directions
            x = hn[-2:,:,:].permute(1,0,2).flatten(start_dim=1)
        elif self.bi_reduce == 'max':
            # Max both directions
            x = torch.max(hn[-2:,:,:], dim=0).values
        elif self.bi_reduce == 'mean':
            # Mean both directions
            x = torch.mean(hn[-2:,:,:], dim=0)
        elif self.bi_reduce == 'last':
            # Just use last direction
            x = hn[-1:,:,:].squeeze(0)

        x = self.fc(x)
        x = self.nl(x)

        return x
            
    def training_step(self, batch, batch_idx):

        if self.lr_scaling:
            # Reset gradients
            opt = self.optimizers()
            opt.zero_grad()

        if self.current_epoch < self.frozen_epochs:
            self.rnn_dp.requires_grad_(False)
        else:
            self.rnn_dp.requires_grad_(True)

        a, b, y = batch

        z_a = self(a)
        z_b = self(b)

        y = y.reshape((-1,1)).float()

        y_hat = self.class_head(z_a, z_b).float()

        loss = self.criterion(y_hat, y)

        # if manhattan, add a distance regularisation
        # we'll have it be 100% of loss, decrease to 0% over time
        if self.class_head_name == 'manhattan':
            d = (z_a - z_b).pow(2)
            indicator = (2*y-1)*-1
            d_reg = max(0, torch.mean(indicator * d))

            delay = 0
            min_contrib = 0.1

            if self.current_epoch > delay:
                interact_alpha = max(self.current_epoch/(self.num_epochs//2), min_contrib)
            else:
                interact_alpha = min_contrib

            reg_alpha = 1 - interact_alpha

            self.log('reg_alpha', reg_alpha)
            self.log('d_reg', d_reg)
            self.log('interact_alpha', interact_alpha)
            self.log('interact_loss', loss)

            loss = reg_alpha * d_reg + loss * interact_alpha

        y_hat_probs = torch.sigmoid(y_hat.flatten().cpu().detach().flatten()).numpy().astype(np.float32)
        y_np = y.flatten().cpu().detach().numpy().astype(int)

        try:
            auroc = roc_auc_score(y_np, y_hat_probs)
        except ValueError as e:
            auroc = -1
        try:
            apr = average_precision_score(y_np, y_hat_probs)
        except ValueError as e:
            apr = -1
        try:
            acc = accuracy_score(y_np, (y_hat_probs > 0.5).astype(int))
        except ValueError as e:
            acc = -1

        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        self.log('train_apr', apr, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.logger.experiment[0].add_pr_curve('train_pr', y, torch.sigmoid(y_hat), self.current_epoch)
            if len(y_hat_probs[y_np == 1]) > 0:
                self.logger.experiment[0].add_histogram('train_pos', y_hat_probs[y_np == 1], self.current_epoch)
                
            if len(y_hat_probs[y_np == 0]) > 0:
                self.logger.experiment[0].add_histogram('train_neg', y_hat_probs[y_np == 0], self.current_epoch)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # When we manually optimise, we lose the loss in the progress bar
        # So that's why prog_bar is tied to lr_scaling
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=self.lr_scaling)

        if self.lr_scaling:

            max_len_a = torch.max(torch.sum(a != 0, axis=1)).item()
            max_len_b = torch.max(torch.sum(b != 0, axis=1)).item()
            total_len = max_len_a + max_len_b

            # LR scaling

            lr_scaled = self.lr_base * total_len / (self.trunc_len * 2)
            opt.param_groups[0]['lr'] = lr_scaled

            self.log('total_seq_len', total_len, on_step=True, on_epoch=False)
            self.log('lr_scaled', lr_scaled, on_step=True, on_epoch=False)
            self.log('lr_base', self.lr_base, on_step=True, on_epoch=False)

            # Propagate losses
            self.manual_backward(loss)
            opt.step()

        return loss
      
    def validation_step(self, batch, batch_idx):
        a, b, y = batch

        z_a = self(a)
        z_b = self(b)

        y = y.reshape((-1,1)).float()

        y_hat = self.class_head(z_a, z_b).float()

        loss = self.criterion(y_hat, y)

        # if manhattan, add a distance regularisation
        # we'll have it be 100% of loss, decrease to 0% over time
        if self.class_head_name == 'manhattan':
            d = (z_a - z_b).pow(2)
            indicator = (2*y-1)*-1
            d_reg = max(0, torch.mean(indicator * d))

            delay = 0
            min_contrib = 0.1

            if self.current_epoch > delay:
                interact_alpha = max(self.current_epoch/(self.num_epochs//2), min_contrib)
            else:
                interact_alpha = min_contrib

            reg_alpha = 1 - interact_alpha

            self.log('val_reg_alpha', reg_alpha)
            self.log('val_d_reg', d_reg)
            self.log('val_interact_alpha', interact_alpha)
            self.log('val_interact_loss', loss)

            loss = reg_alpha * d_reg + loss * interact_alpha

        y_hat_probs = torch.sigmoid(y_hat.flatten().cpu().detach()).numpy().astype(np.float32)
        y_np = y.flatten().cpu().detach().numpy().astype(int)

        try:
            auroc = roc_auc_score(y_np, y_hat_probs)
        except ValueError as e:
            auroc = -1

        try:
            apr = average_precision_score(y_np, y_hat_probs)
        except ValueError as e:
            apr = -1

        try:
            acc = accuracy_score(y_np, (y_hat_probs > 0.5).astype(int))
        except ValueError as e:
            acc = -1

        self.log('val_auroc', auroc)
        self.log('val_apr', apr)
        self.log('val_acc', acc, prog_bar=True)

        if batch_idx == 0:
            self.logger.experiment[0].add_pr_curve('val_pr', y, torch.sigmoid(y_hat), self.current_epoch)

            if len(y_hat_probs[y_np == 1]) > 0:
                self.logger.experiment[0].add_histogram('val_pos', y_hat_probs[y_np == 1], self.current_epoch)

            if len(y_hat_probs[y_np == 0]) > 0:
                self.logger.experiment[0].add_histogram('val_neg', y_hat_probs[y_np == 0], self.current_epoch)

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        a, b, y = batch

        z_a = self(a)
        z_b = self(b)

        y = y.reshape((-1,1)).float()

        y_hat = self.class_head(z_a, z_b).float()

        loss = self.criterion(y_hat, y)

        # if manhattan, add a distance regularisation
        # we'll have it be 100% of loss, decrease to 0% over time
        if self.class_head_name == 'manhattan':
            d = (z_a - z_b).pow(2)
            indicator = (2*y-1)*-1
            d_reg = max(0, torch.mean(indicator * d))

            delay = 0
            min_contrib = 0.1

            if self.current_epoch > delay:
                interact_alpha = max(self.current_epoch/(self.num_epochs//2), min_contrib)
            else:
                interact_alpha = min_contrib

            reg_alpha = 1 - interact_alpha

            loss = reg_alpha * d_reg + loss * interact_alpha

        y_hat_probs = torch.sigmoid(y_hat.flatten().cpu().detach()).numpy().astype(np.float32)
        y_np = y.flatten().cpu().detach().numpy().astype(int)

        try:
            auroc = roc_auc_score(y_np, y_hat_probs)
        except ValueError as e:
            auroc = -1

        try:
            apr = average_precision_score(y_np, y_hat_probs)
        except ValueError as e:
            apr = -1

        try:
            acc = accuracy_score(y_np, (y_hat_probs > 0.5).astype(int))
        except ValueError as e:
            acc = -1

        self.log('test_auroc', auroc)
        self.log('test_apr', apr)
        self.log('test_acc', acc, prog_bar=True)

        if batch_idx == 0:
            self.logger.experiment[0].add_pr_curve('test_pr', y, torch.sigmoid(y_hat), self.current_epoch)

            if len(y_hat_probs[y_np == 1]) > 0:
                self.logger.experiment[0].add_histogram('test_pos', y_hat_probs[y_np == 1], self.current_epoch)

            if len(y_hat_probs[y_np == 0]) > 0:
                self.logger.experiment[0].add_histogram('test_neg', y_hat_probs[y_np == 0], self.current_epoch)

        self.log('test_loss', loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):

        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.optimizer_type == 'ranger21':
            optimizer = Ranger21(self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, 
            num_batches_per_epoch=self.steps_per_epoch, 
            num_epochs=self.num_epochs,
            warmdown_start_pct=0.72)
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return optimizer