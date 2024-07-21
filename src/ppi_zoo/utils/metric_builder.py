from torch import nn

from ppi_zoo.metrics.MetricModule import MetricModule
from ppi_zoo.metrics.MetricEnum import METRIC_ENUM

def build_metrics(nr_dataloaders: int) -> nn.ModuleList:
    metrics_list = []
    for key, value in METRIC_ENUM.items():
        for dataloader_idx in range(0, nr_dataloaders):
            metrics_list.append(
                MetricModule(
                    name=key,
                    metric=value.get('metric'),
                    dataloader_idx=dataloader_idx,
                    log=value.get('log')
                )
            )
    return nn.ModuleList(metrics_list)