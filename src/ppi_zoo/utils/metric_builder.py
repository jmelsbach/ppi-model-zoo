from torch import nn

from ppi_zoo.metrics.MetricModule import MetricModule
from ppi_zoo.metrics.MetricEnum import METRIC_ENUM

def build_metrics(nr_dataloaders: int) -> nn.ModuleList:
    metrics_list = []
    for key, value in METRIC_ENUM.items():
        build_metric = value.get('build_metric')
        log = value.get('log')
        for dataloader_idx in range(0, nr_dataloaders):
            metrics_list.append(
                MetricModule(
                    name=key,
                    metric=build_metric(),
                    dataloader_idx=dataloader_idx,
                    log=log
                )
            )
    return nn.ModuleList(metrics_list)