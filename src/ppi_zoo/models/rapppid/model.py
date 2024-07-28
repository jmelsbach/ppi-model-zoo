import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, precision_recall_curve
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

from ranger21 import Ranger21
from weightdrop import WeightDrop
from nl import Mish

class MeanClassHead(nn.Module):
    def __init__(self, embedding_size, num_layers, weight_drop, variational):
        super(MeanClassHead, self).__init__()#

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
        self.embedding_size = embedding_size
        self.lstm_dropout_rate = lstm_dropout_rate
        self.classhead_dropout_rate = classhead_dropout_rate
        self.rnn_num_layers = rnn_num_layers
        self.classhead_num_layers = classhead_num_layers
        self.lr_base = lr
        self.embedding_droprate = embedding_droprate
        self.weight_decay = weight_decay
        self.bi_reduce = bi_reduce
        self.class_head_name = class_head_name
        self.frozen_epochs = frozen_epochs
        self.optimizer_type = optimizer_type
        
        # Define layers and components
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.nl = Mish()

        # Define RNN
        if self.bi_reduce == 'concat':
            self.rnn = nn.LSTM(embedding_size, embedding_size // 2, rnn_num_layers, bidirectional=True, batch_first=True)
        elif self.bi_reduce in ['max', 'mean', 'last']:
            self.rnn = nn.LSTM(embedding_size, embedding_size, rnn_num_layers, bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Unexpected value for `bi_reduce`: {bi_reduce}")

        self.rnn_dp = WeightDrop(self.rnn, ['weight_hh_l0'], lstm_dropout_rate, variational_dropout)

        # Define class head
        if class_head_name == 'concat':
            self.class_head = ConcatClassHead(embedding_size, classhead_num_layers, classhead_dropout_rate, variational_dropout)
        elif class_head_name == 'mean':
            self.class_head = MeanClassHead(embedding_size, classhead_num_layers, classhead_dropout_rate, variational_dropout)
        elif class_head_name == 'mult':
            self.class_head = MultClassHead(embedding_size, classhead_num_layers, classhead_dropout_rate, variational_dropout)
        elif class_head_name == 'manhattan':
            self.class_head = ManhattanClassHead()
        else:
            raise ValueError(f"Unexpected value for `class_head_name`: {class_head_name}")

        self.criterion = nn.BCEWithLogitsLoss()
        self.embedding = nn.Embedding(num_codes, embedding_size, padding_idx=0)

        # Manual optimization if lr_scaling is enabled
        if lr_scaling:
            self.automatic_optimization = False

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

    def single_step(self, batch):
        a, b, targets = batch
        z_a = self(a)
        z_b = self(b)
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

            self.log('reg_alpha', reg_alpha)
            self.log('d_reg', d_reg)
            self.log('interact_alpha', interact_alpha)
            self.log('interact_loss', loss)

            loss = reg_alpha * d_reg + loss * interact_alpha

        return loss, predictions, targets

    def training_step(self, batch, batch_idx):
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

        loss, predictions, targets = self.single_step(batch)

        # Compute metrics
        predictions_probs = torch.sigmoid(predictions.flatten()).cpu().detach().numpy().astype(np.float32)
        targets_np = targets.flatten().cpu().detach().numpy().astype(int)

        try:
            auroc = roc_auc_score(targets_np, predictions_probs)
        except ValueError:
            auroc = -1
        try:
            apr = average_precision_score(targets_np, predictions_probs)
        except ValueError:
            apr = -1
        try:
            acc = accuracy_score(targets_np, (predictions_probs > 0.5).astype(int))
        except ValueError:
            acc = -1

        # Log metrics
        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        self.log('train_apr', apr, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.logger.experiment[0].add_pr_curve('train_pr', targets, torch.sigmoid(predictions), self.current_epoch)
            if len(predictions_probs[targets_np == 1]) > 0:
                self.logger.experiment[0].add_histogram('train_pos', predictions_probs[targets_np == 1], self.current_epoch)
            if len(predictions_probs[targets_np == 0]) > 0:
                self.logger.experiment[0].add_histogram('train_neg', predictions_probs[targets_np == 0], self.current_epoch)

        # Log loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=self.lr_scaling)

        if self.lr_scaling:
            # Adjust learning rate based on sequence lengths
            max_len_a = torch.max(torch.sum(a != 0, axis=1))
            max_len_b = torch.max(torch.sum(b != 0, axis=1))
            new_lr = self.lr_base / (max_len_a + max_len_b)
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            # Backpropagation
            self.manual_backward(loss)
            opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, targets = self.single_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, predictions, targets = self.single_step(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
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