import torch
from torch import nn
from ppi_zoo.models.rapppid.Mish import Mish

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