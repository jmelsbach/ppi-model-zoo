from torch import nn

from ppi_zoo.metrics.MetricModule import MetricModule
from ppi_zoo.metrics.MetricEnum import METRIC_ENUM

LOG_KEYS = {
    'fit': 'F',
    'validate': 'V',
    'test': 'T'
}

def build_metric_title(name, dataloader_idx, stage):
    if dataloader_idx is None:
        return name
    
    return f'{name} {LOG_KEYS[stage]}{dataloader_idx + 1}'

def build_metric_log_key(name, dataloader_idx, stage):
    if dataloader_idx is None:
        return f'{stage}_{name}'
    
    return f'{name}_{LOG_KEYS[stage]}{dataloader_idx + 1}'

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
                    dataloader_idx=dataloader_idx if nr_dataloaders > 1 else None,
                    log=log
                )
            )
    return nn.ModuleList(metrics_list)
