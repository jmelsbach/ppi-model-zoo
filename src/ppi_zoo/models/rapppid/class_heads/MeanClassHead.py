from torch import nn
from ppi_zoo.models.rapppid.Mish import Mish
from ppi_zoo.models.rapppid.WeightDrop import WeightDrop

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