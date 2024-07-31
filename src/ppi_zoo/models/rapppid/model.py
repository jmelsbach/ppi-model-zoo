import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

from ppi_zoo.utils.metric_builder import build_metrics
from ppi_zoo.metrics.MetricModule import MetricModule
from ranger21 import Ranger21
from weightdrop import WeightDrop


class MeanClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MeanClassHead, self).__init__()

        if num_layers == 1:
            self.fc = WeightDrop(nn.Linear(embedding_size, 1), ['weight'],
                                    dropout=weight_drop, variational=variational)
        elif num_layers == 2:
            self.fc = nn.Sequential(
                        nn.Linear(embedding_size, embedding_size//2),
                        nn.Linear(embedding_size//2, 1))
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        z = (z_a + z_b)/2
        z = self.fc(z)
        if hasattr(self, 'nl'):
            z = z * torch.tanh(F.softplus(z))
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
                        WeightDrop(nn.Linear(embedding_size//2, 1), 
                                    ['weight'], dropout=weight_drop, 
                                    variational=variational)
                                    )
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        z_a = (z_a - z_a.mean()) / z_a.std()
        z_b = (z_b - z_b.mean()) / z_b.std()
        z = z_a * z_b
        z = z * torch.tanh(F.softplus(z))
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
                        nn.Linear(embedding_size//2, 1))
        else:
            raise NotImplementedError

    def forward(self, z_a, z_b):
        z_ab = torch.cat((z_a, z_b), axis=1)
        z = self.fc(z_ab)
        if hasattr(self, 'nl'):
            z = z * torch.tanh(F.softplus(z))
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
                 classhead_num_layers, lr, weight_decay, bi_reduce, 
                 class_head_name, variational_dropout, lr_scaling, trunc_len, 
                 embedding_droprate, frozen_epochs, optimizer_type):
        super(LSTMAWD, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize parameters
        self.lr_scaling = lr_scaling
        self.trunc_len = trunc_len
        self.num_epochs = num_epochs
        self.lr_base = lr
        self.embedding_droprate = embedding_droprate
        self.weight_decay = weight_decay
        self.frozen_epochs = frozen_epochs
        self.optimizer_type = optimizer_type

        # Initialize model build parameters
        self.embedding_size = embedding_size
        self.bi_reduce = bi_reduce
        self.rnn_num_layers = rnn_num_layers
        self.lstm_dropout_rate = lstm_dropout_rate
        self.class_head_name = class_head_name
        self.classhead_num_layers = classhead_num_layers
        self.classhead_dropout_rate = classhead_dropout_rate
        self.variational_dropout = variational_dropout
        self.num_codes = num_codes

        self._build_model()

        # Manual optimization if lr_scaling is enabled
        if lr_scaling:
            self.automatic_optimization = False

    def _build_model(self) -> None:
        # Define layers and components
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)
        self.nl = lambda x: x * torch.tanh(F.softplus(x))

        # Define RNN
        if self.bi_reduce == 'concat':
            self.rnn = nn.LSTM(self.embedding_size, self.embedding_size // 2, self.rnn_num_layers, bidirectional=True, batch_first=True)
        elif self.bi_reduce in ['max', 'mean', 'last']:
            self.rnn = nn.LSTM(self.embedding_size, self.embedding_size, self.rnn_num_layers, bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Unexpected value for `bi_reduce`: {self.bi_reduce}")

        self.rnn_dp = WeightDrop(self.rnn, ['weight_hh_l0'], self.lstm_dropout_rate, self.variational_dropout)

        # Define class head
        if self.class_head_name == 'concat':
            self.class_head = ConcatClassHead(self.embedding_size, self.classhead_num_layers, self.classhead_dropout_rate, self.variational_dropout)
        elif self.class_head_name == 'mean':
            self.class_head = MeanClassHead(self.embedding_size, self.classhead_num_layers, self.classhead_dropout_rate, self.variational_dropout)
        elif self.class_head_name == 'mult':
            self.class_head = MultClassHead(self.embedding_size, self.classhead_num_layers, self.classhead_dropout_rate, self.variational_dropout)
        elif self.class_head_name == 'manhattan':
            self.class_head = ManhattanClassHead()
        else:
            raise ValueError(f"Unexpected value for `class_head_name`: {self.class_head_name}")

        self.criterion = nn.BCEWithLogitsLoss()
        self.embedding = nn.Embedding(self.num_codes, self.embedding_size, padding_idx=0)
    
    # TODO: Hat das keine Vorimplementierung in pytorch oder lightning?
    def embedding_dropout(self, embed, words, p=0.2):
        """
        Apply dropout to the embedding layer.
        
        Args:
            embed (nn.Embedding): The embedding layer.
            words (torch.Tensor): Input tensor containing word indices.
            p (float): Dropout probability.

        Returns:
            torch.Tensor: Embedding tensor with dropout applied.
        """
        if not self.training or p == 0:
            return F.embedding(words, embed.weight, embed.padding_idx, embed.max_norm, embed.norm_type,
                            embed.scale_grad_by_freq, embed.sparse)

        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
        masked_embed_weight = mask * embed.weight

        return F.embedding(words, masked_embed_weight, embed.padding_idx, embed.max_norm, embed.norm_type,
                        embed.scale_grad_by_freq, embed.sparse)

    def forward(self, x):
        # Truncate to the longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x = self.embedding_dropout(self.embedding, x, p=self.embedding_droprate)
        output, (hn, cn) = self.rnn_dp(x)

        if self.bi_reduce == 'concat':
            # Concat both directions
            x = hn[-2:, :, :].permute(1, 0, 2).reshape(hn.size(1), -1)
        elif self.bi_reduce == 'max':
            # Max both directions
            x = torch.max(hn[-2:, :, :], dim=0).values
        elif self.bi_reduce == 'mean':
            # Mean both directions
            x = torch.mean(hn[-2:, :, :], dim=0)
        elif self.bi_reduce == 'last':
            # Just use last direction
            x = hn[-1, :, :]

        x = self.fc(x)
        x = self.nl(x)

        return x



    def _single_step(self, batch):
        inputs_A, inputs_B, targets = batch
        z_a = self(inputs_A)
        z_b = self(inputs_B)
        targets = targets.reshape((-1, 1)).float()
        predictions = self.class_head(z_a, z_b).float()
        loss = self.criterion(predictions, targets)
        
        if self.class_head_name == 'manhattan':
            d = (z_a - z_b).pow(2)
            indicator = (2 * targets - 1) * -1
            d_reg = max(0, torch.mean(indicator * d))
            delay = 0
            min_contrib = 0.1
            if self.current_epoch > delay:
                interact_alpha = max(self.current_epoch / (self.num_epochs // 2), min_contrib)
            else:
                interact_alpha = min_contrib
            reg_alpha = 1 - interact_alpha
            loss = reg_alpha * d_reg + loss * interact_alpha

        return targets, predictions, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # Access optimizer
        opt = self.optimizers() if self.lr_scaling else None
        
        if self.lr_scaling:
            # Reset gradients
            opt.zero_grad()

        # Freeze/Unfreeze layers based on current epoch
        if self.current_epoch < self.frozen_epochs:
            self.rnn_dp.requires_grad_(False)
        else:
            self.rnn_dp.requires_grad_(True)

        targets, predictions, train_loss = self._single_step(batch)

        # Log loss
        self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        self.log('train_loss_step', train_loss, on_step=True, on_epoch=False, prog_bar=self.lr_scaling)

        if self.lr_scaling:
            # Adjust learning rate based on sequence lengths
            max_len_a = torch.max(torch.sum(a != 0, axis=1))
            max_len_b = torch.max(torch.sum(b != 0, axis=1))
            new_lr = self.lr_base / (max_len_a + max_len_b)
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            # Backpropagation
            self.manual_backward(train_loss)
            opt.step()

        return train_loss
    
    # metrics functions
    def _update_metrics(self, predictions, targets, metric_modules: list) -> None:
        metric_module: MetricModule
        for metric_module in metric_modules:
            metric_module.metric.update(predictions, targets)
    
    def _log_metrics(self, stage: str) -> None:
        metric_module: MetricModule
        for metric_module in self._metrics:
            if metric_module.log:
                metric_module.log(self.logger, metric_module.metric, metric_module.dataloader_idx)
                continue
            
            key = f'{stage}_{metric_module.name}'
            if metric_module.dataloader_idx:
                key = f'{key}_{metric_module.dataloader_idx}'
            self.log(key, metric_module.metric.compute())

    def _reset_metrics(self) -> None:
        metric_module: MetricModule
        for metric_module in self._metrics:
            metric_module.metric.reset()

    # validation and testing
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        targets, predictions, val_loss = self._single_step(batch)

        self.log(f'val_loss', val_loss)
        self._update_metrics(
            predictions,
            targets,
            filter(lambda metric: metric.dataloader_idx == dataloader_idx, self._metrics)
        )

        return val_loss
    
    def on_validation_epoch_end(self) -> None:
        self._log_metrics('val')
        self._reset_metrics()

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        targets, predictions, test_loss = self._single_step(batch)
        self.log(f'test_loss', test_loss)
        self._update_metrics(
            predictions,
            targets,
            filter(lambda metric: metric.dataloader_idx == dataloader_idx, self._metrics)
        )
    
    def on_test_epoch_end(self) -> None:
        self._log_metrics('test')
        self._reset_metrics()
    
    def configure_optimizers(self) -> tuple:
        if self.optimizer_type == 'ranger21':
            optimizer = Ranger21(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                num_batches_per_epoch=self.steps_per_epoch, 
                num_epochs=self.num_epochs,
                warmdown_start_pct=0.72
            )
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return optimizer