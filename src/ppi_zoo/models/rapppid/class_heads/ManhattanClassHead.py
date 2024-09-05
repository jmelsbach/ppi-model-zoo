import torch
from torch import nn

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
