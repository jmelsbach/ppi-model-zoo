from torch import nn


class MetricModule(nn.Module):
    def __init__(self, name, metric, dataloader_idx=None, log=None):
        super(MetricModule, self).__init__()
        self.name = name
        self.metric = metric
        self.dataloader_idx = dataloader_idx
        self.log = log

    def __str__(self) -> str:
        return f'Name: {self.name}; Metric: {self.metric}; Dataloader idx: {self.dataloader_idx}; Log: {self.log}'