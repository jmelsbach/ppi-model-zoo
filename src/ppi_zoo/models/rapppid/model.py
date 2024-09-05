import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import lightning.pytorch as L
from ranger21 import Ranger21

from ppi_zoo.models.GoldStandardPPILightningModule import GoldStandardPPILightningModule
from ppi_zoo.models.rapppid.Mish import Mish
from ppi_zoo.models.rapppid.WeightDrop import WeightDrop
from ppi_zoo.models.rapppid.class_heads.MeanClassHead import MeanClassHead
from ppi_zoo.models.rapppid.class_heads.MultClassHead import MultClassHead
from ppi_zoo.models.rapppid.class_heads.ConcatClassHead import ConcatClassHead
from ppi_zoo.models.rapppid.class_heads.ManhattanClassHead import ManhattanClassHead

class LSTMAWD(GoldStandardPPILightningModule):
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
        optimizer_type: str = 'ranger21' 
    ) -> None:
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        
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

        self._build_model()

        # Manual optimization if lr_scaling is enabled
        if lr_scaling:
            self.automatic_optimization = False

    def setup(self, stage=None):
        super().setup(stage)
        datamodule: L.LightningDataModule = self.trainer.datamodule
        self.steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch else len(datamodule.train_dataloader())
    
    def forward(self, seq_A: torch.Tensor, seq_B: torch.Tensor) -> torch.Tensor:
        z_a = self._reduce(seq_A) 
        z_b = self._reduce(seq_B)
        
        return self.class_head(z_a, z_b).squeeze(-1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if self.lr_scaling:
            # Reset gradients
            opt = self.optimizers()
            opt.zero_grad()

        # Freeze/Unfreeze layers based on current epoch
        if self.current_epoch < self.frozen_epochs:
            self.rnn_dp.requires_grad_(False)
        else:
            self.rnn_dp.requires_grad_(True)

        train_loss = super().training_step(batch, batch_idx)
        
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

        # Deine embedding
        self.embedding = nn.Embedding(num_embeddings = self.num_codes, embedding_dim = self.embedding_size, padding_idx=0) # embeddings for each token are learned in the model training -> simple lookup table that stores embeddings of a fixed dictionary and size.
    
    def _embedding_dropout(self, embed, words, p=0.2):
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

        x = self._embedding_dropout(self.embedding, x, p=self.embedding_droprate) # creates torch.Size([16, 701, 64])
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
