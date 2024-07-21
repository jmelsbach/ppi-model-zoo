from torch import nn


class MetricModule(nn.Module):
    def __init__(self, name, metric, dataloader_idx=None, log=None):
        super(MetricModule, self).__init__()
        self.name = name
        self.metric = metric
        self.dataloader_idx = dataloader_idx
        self.log = log