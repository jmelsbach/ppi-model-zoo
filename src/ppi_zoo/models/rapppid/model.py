import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import lightning.pytorch as L
import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, precision_recall_curve

from ppi_zoo.utils.metric_builder import build_metrics
from ppi_zoo.metrics.MetricModule import MetricModule
from ranger21 import Ranger21
    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))
    
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
        z = self.fc(z) # todo: check if we apply sigmoid here? -> is that maybe done by the 

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
    def __init__(self):
        super(ManhattanClassHead, self).__init__()

        self.fc = nn.Linear(1, 1)

    def forward(self, z_a, z_b):
        
        distance = torch.sum(torch.abs(z_a-z_b), dim=1).unsqueeze(1)
        y_logit = self.fc(distance)

        return y_logit


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=True):
        """
        Dropout class that is paired with a torch module to make sure that the SAME mask
        will be sampled and applied to ALL timesteps.
        :param module: nn. module (e.g. nn.Linear, nn.LSTM)
        :param weights: which weights to apply dropout (names of weights of module)
        :param dropout: dropout to be applied
        :param variational: if True applies Variational Dropout, if False applies DropConnect (different masks!!!)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        """
        Smerity code I don't understand.
        """
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        """
        This function renames each 'weight name' to 'weight name' + '_raw'
        (e.g. weight_hh_l0 -> weight_hh_l0_raw)
        :return:
        """
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        """
        This function samples & applies a dropout mask to the weights of the recurrent layers.
        Specifically, for an LSTM, each gate has
        - a W matrix ('weight_ih') that is multiplied with the input (x_t)
        - a U matrix ('weight_hh') that is multiplied with the previous hidden state (h_t-1)
        We sample a mask (either with Variational Dropout or with DropConnect) and apply it to
        the matrices U and/or W.
        The matrices to be dropped-out are in self.weights.
        A 'weight_hh' matrix is of shape (4*nhidden, nhidden)
        while a 'weight_ih' matrix is of shape (4*nhidden, ninput).
        **** Variational Dropout ****
        With this method, we sample a mask from the tensor (4*nhidden, 1) PER ROW
        and expand it to the full matrix.
        **** DropConnect ****
        With this method, we sample a mask from the tensor (4*nhidden, nhidden) directly
        which means that we apply dropout PER ELEMENT/NEURON.
        :return:
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None

            if self.variational:
                #######################################################
                # Variational dropout (as proposed by Gal & Ghahramani)
                #######################################################
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                #######################################################
                # DropConnect (as presented in the AWD paper)
                #######################################################
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)

            if not self.training: # (*)
                w = w.data
                
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class LSTMAWD(L.LightningModule):
    def __init__(
        self,
        num_codes: int = 250,
        embedding_size: int = 64,
        steps_per_epoch: int = None,
        num_epochs: int = 100,
        lstm_dropout_rate: float = 0.3,
        classhead_dropout_rate: float = 0.2,
        rnn_num_layers: int = 2,
        classhead_num_layers: int = 2,
        lr: float = 0.01,
        weight_decay: float = 0.0001,
        bi_reduce: str = 'last',
        class_head_name: str = 'mult',
        variational_dropout: bool = False,
        lr_scaling: bool = False,
        trunc_len: int = 1500,
        embedding_droprate: float = 0.3,
        frozen_epochs: int = 0,
        optimizer_type: str = 'adam' # todo: originally rapppid implements 'ranger21'
    ) -> None:
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        #self.hparams.lr_scaling -> Zugriff ohne Initialisierung
        
        # Initialize training parameters
        self.lr_scaling = lr_scaling
        self.trunc_len = trunc_len
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_base = lr
        self.lr = lr
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

        self._build_model() # creates an LSTM model and wraps this by WeightDrop

        # Manual optimization if lr_scaling is enabled
        if lr_scaling:
            self.automatic_optimization = False

        self.criterion = nn.BCEWithLogitsLoss()
    
    def setup(self, stage=None): 
        datamodule = self.trainer.datamodule
        # todo: add this to model
        # todo: check if modulo is wanted here!
        self.steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch else len(datamodule.train_dataloader())
        nr_dataloaders = 1
        if stage == 'fit' or stage == 'validate':
            nr_dataloaders = len(datamodule.val_dataloader()) if type(datamodule.val_dataloader()) is list else 1
            
        if stage == 'test' or stage is None:
            nr_dataloaders = len(datamodule.test_dataloader()) if type(datamodule.test_dataloader()) is list else 1
        
        self._metrics = build_metrics(nr_dataloaders)

    def _build_model(self) -> None:
        # Define layers and components
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)
        self.nl = Mish()
        # Define Recurrent Neural Network
        if self.bi_reduce not in ['concat', 'max', 'mean', 'last']:
            raise ValueError(f"Unexpected value for `bi_reduce`: {self.bi_reduce}")
        rnn_hidden_size = self.embedding_size // 2 if self.bi_reduce == 'concat' else self.embedding_size
        self.rnn = nn.LSTM(self.embedding_size, rnn_hidden_size, self.rnn_num_layers, bidirectional=True, batch_first=True)
        self.rnn_dp = WeightDrop(self.rnn, ['weight_hh_l0'], self.lstm_dropout_rate, self.variational_dropout)

        # Define classificstion head
        class_head_map = {
            'concat': ConcatClassHead,
            'mean': MeanClassHead,
            'mult': MultClassHead,
            'manhattan': ManhattanClassHead
        }
        if self.class_head_name not in class_head_map:
            raise ValueError(f"Unexpected value for `class_head_name`: {self.class_head_name}")

        self.class_head = class_head_map[self.class_head_name](
            self.embedding_size, self.classhead_num_layers, self.classhead_dropout_rate, self.variational_dropout
        )

        self.embedding = nn.Embedding(num_embeddings = self.num_codes, embedding_dim = self.embedding_size, padding_idx=0) # embeddings for each token are learned in the model training -> simple lookup table that stores embeddings of a fixed dictionary and size.
    
    def embedding_dropout(self, embed, words, p=0.2):
        """
        Apply dropout to the embedding layer. 
        The embedding dropout layer randomly assigns random tokens from the total vocabulary to zero
        
        Args:
            embed (nn.Embedding): The embedding layer. It is a matrix where each row corresponds to a vector representation (embedding) for a token.
            words (torch.Tensor): Input tensor containing word indices.
            p (float): Dropout probability.

        Returns:
            torch.Tensor: Embedding tensor with dropout applied.
        """

        padding_idx = embed.padding_idx or -1
        if not self.training or p == 0:
            embed_weight = embed.weight
        else:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
            embed_weight = mask * embed.weight

        return F.embedding(words, embed_weight, padding_idx, embed.max_norm, embed.norm_type, # this creates an embedding for each token with size 64
                        embed.scale_grad_by_freq, embed.sparse)
    
    def _reduce(self, x) -> torch.Tensor:
        # Truncate to the longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1)) # aminosäure = "ABACAD" -> x = [1,2, 6, ..., 0, 0, 0, 0]
        x = x[:, :max_len] # reduces shape from torch.Size([16, 1000]) to torch.Size([16, 756]) Note: can also be different size as 756

        x = self.embedding_dropout(self.embedding, x, p=self.embedding_droprate) # creates torch.Size([16, 701, 64])
        output, (hn, cn) = self.rnn_dp(x) # hn.shape [4, batch_size, embeddings_size] -> This dimension represents the number of layers multiplied by the number of directions in the LSTM. Given that we have an LSTM with 2 layers and it’s bidirectional, this dimension is 2 (layers) * 2 (directions) = 4.

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

    def forward(self, inputs_A, inputs_B) -> torch.Tensor:
        targets = targets.reshape((-1, 1)).float()
        predictions = self.class_head(self._reduce(inputs_A), self._reduce(inputs_B)).float()
        return predictions
    
    def _calculate_loss(self, z_a, z_b, predictions, targets):
        # Compute standard BCE loss
        loss = self.criterion(predictions, targets)
        
        if self.class_head_name == 'manhattan': # todo: can be deleted is not used!
            # Compute distance regularization term
            d = (z_a - z_b).pow(2)  # Squared differences
            indicator = (2 * targets - 1) * -1  # Indicator for regularization
            d_reg = max(0, torch.mean(indicator * d))
            interact_alpha = max(self.current_epoch / (self.num_epochs // 2), 0.1) if self.current_epoch > 0 else 0.1
            reg_alpha = 1 - interact_alpha
            loss = reg_alpha * d_reg + loss * interact_alpha

        return loss

    def _single_step(self, batch):
        inputs_A, inputs_B, targets = batch # inputs.shape = torch.Size([16, 1000]) = (batch, sequence representation)
        targets = targets.unsqueeze(-1).float() #.reshape((-1,1)).float()
        # Get embeddings
        z_a = self._reduce(inputs_A) 
        z_b = self._reduce(inputs_B)
        
        # Get predictions
        predictions = self.class_head(z_a, z_b).float()
        
        # Calculate loss
        loss = self._calculate_loss(z_a, z_b, predictions, targets)
        return targets, predictions, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        if self.lr_scaling:
            # Reset gradients
            opt = self.optimizers()
            opt.zero_grad()

        # Freeze/Unfreeze layers based on current epoch
        if self.current_epoch < self.frozen_epochs:
            self.rnn_dp.requires_grad_(False)
        else:
            self.rnn_dp.requires_grad_(True)

        _, _, train_loss = self._single_step(batch)

        self.log('train_loss', train_loss)
        
        if self.lr_scaling:
            inputs_A, inputs_B, _ = batch
            # Adjust learning rate based on sequence lengths
            max_len_a = torch.max(torch.sum(inputs_A != 0, axis=1)).item()
            max_len_b = torch.max(torch.sum(inputs_B != 0, axis=1)).item()
            new_lr = self.lr_base * (max_len_a + max_len_b) / (self.trunc_len * 2)
            opt.param_groups[0]['lr'] = new_lr

            # Backpropagation
            self.manual_backward(train_loss)
            opt.step()

        return train_loss
    
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
    
    # optimizer
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
    
    # metrics functions
    def _update_metrics(self, predictions, targets, metric_modules: list):
        metric_module: MetricModule
        for metric_module in metric_modules:
            metric_module.metric.update(predictions, targets.int())
        
    def _log_metrics(self, stage: str):
        self._debug_print(f'Logging metrics for stage: {stage}')
        metric_module: MetricModule
        for metric_module in self._metrics:
            if metric_module.log:
                metric_module.log(self, metric_module.metric, metric_module.dataloader_idx)
                continue
            
            key = f'{stage}_{metric_module.name}'
            
            if metric_module.dataloader_idx is not None:
                key = f'{key}_T{metric_module.dataloader_idx + 1}'

            value = metric_module.metric.compute()

            self._debug_print(f'Metric {metric_module.name} scored value of {value} on T{metric_module.dataloader_idx + 1} ')
            self.log(key, value, sync_dist=True)

    def _reset_metrics(self):
        metric_module: MetricModule
        for metric_module in self._metrics:
            metric_module.metric.reset()

    def _debug_print(self, message):
        if self.global_rank != 0:
            return
        
        print(message)
