import lightning.pytorch as L
import torch
from torch import nn
from typing import Tuple, List
from ppi_zoo.utils.metrics import build_metrics, build_metric_log_key
from ppi_zoo.metrics.MetricModule import MetricModule


class GoldStandardPPILightningModule(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.loss_function = nn.BCEWithLogitsLoss()

    def setup(self, stage=None):
        datamodule: L.LightningDataModule = self.trainer.datamodule
        nr_dataloaders = 1
        if stage == 'fit' or stage == 'validate':
            nr_dataloaders = len(datamodule.val_dataloader()) if type(datamodule.val_dataloader()) is list else 1
            
        if stage == 'test' or stage is None:
            nr_dataloaders = len(datamodule.test_dataloader()) if type(datamodule.test_dataloader()) is list else 1
        
        self.nr_dataloaders: int = nr_dataloaders
        self._metrics: List[MetricModule] = build_metrics(self.nr_dataloaders)

    def _build_model():
        raise NotImplementedError

    def forward(seq_A: torch.Tensor, seq_B: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def _single_step(self, batch:list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_A, seq_B, targets = batch
        targets = targets.unsqueeze(-1).float()
        predictions = self.forward(seq_A, seq_B)
        loss = self.loss_function(
            predictions,
            targets
        )
        return targets, predictions, loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, _, train_loss = self._single_step(batch)

        self.log('train_loss', train_loss, sync_dist=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        targets, predictions, val_loss = self._single_step(batch)

        self.log(f'validate_loss', val_loss, sync_dist=True)
        self._update_metrics(
            predictions,
            targets,
            filter(lambda metric: metric.dataloader_idx == dataloader_idx, self._metrics)
        )

        return val_loss

    def on_validation_epoch_end(self):
        self._log_metrics('validate')
        self._reset_metrics()

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        targets, predictions, test_loss = self._single_step(batch)

        self.log(f'test_loss', test_loss, sync_dist=True)
        self._update_metrics(
            predictions,
            targets,
            filter(lambda metric: metric.dataloader_idx == dataloader_idx, self._metrics)
        )

        return test_loss

    def on_test_epoch_end(self):
        self._log_metrics('test')
        self._reset_metrics()

    def _update_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, metric_modules: List[MetricModule]):
        for metric_module in metric_modules:
            metric_module.metric.update(predictions, targets.int())

    def _log_metrics(self, stage: str):
        self._debug_print(f'Logging metrics for stage: {stage}')
        for metric_module in self._metrics:
            dataloader_idx: int = metric_module.dataloader_idx if self.nr_dataloaders > 1 else None # TODO: if nr_dataloaders is 1 then dataloader_idx for all metrics should be None -> change default value of dataloader_idx in training_step etc.
            if metric_module.log:
                metric_module.log(self, metric_module.metric, dataloader_idx, stage)
                continue

            key = build_metric_log_key(metric_module.name, dataloader_idx, stage)
            value = metric_module.metric.compute()

            self._debug_print(f'Metric {key} scored value of {value}')
            self.log(key, value, sync_dist=True)

    def _reset_metrics(self):
        metric_module: MetricModule
        for metric_module in self._metrics:
            metric_module.metric.reset()

    def _debug_print(self, message: str):
        if self.global_rank != 0:
            return
        
        print(message)

