from torch import nn
from torchmetrics.metric import Metric

class MetricModule(nn.Module):
    def __init__(self, name, metric, dataloader_idx=None, log=None):
        super(MetricModule, self).__init__()
        self.name:str = name
        self.metric: Metric = metric
        self.dataloader_idx:int = dataloader_idx
        self.log:function = log

    def __str__(self) -> str:
        return f'Name: {self.name}; Metric: {self.metric}; Dataloader idx: {self.dataloader_idx}; Log: {self.log}'