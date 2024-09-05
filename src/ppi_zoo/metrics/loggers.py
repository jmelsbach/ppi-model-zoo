import lightning.pytorch as L
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule
from torchmetrics import (
    ConfusionMatrix,
    PrecisionRecallCurve,
    ROC,
)
import ppi_zoo.utils.metrics as metric_util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_dataset_balance(module: LightningModule, dataloader_idx: int, stage: str) -> float:
    datamodule: L.LightningDataModule = module.trainer.datamodule
    dataloader: DataLoader = None
    if stage == 'test':
        dataloader = datamodule.test_dataloader()[dataloader_idx] if type(datamodule.test_dataloader()) is list else datamodule.test_dataloader()
    elif stage == 'fit':
        dataloader = datamodule.train_dataloader()[dataloader_idx] if type(datamodule.train_dataloader()) is list else datamodule.train_dataloader()
    elif stage == 'validate':
        dataloader = datamodule.val_dataloader()[dataloader_idx] if type(datamodule.val_dataloader()) is list else datamodule.val_dataloader()

    pos_count: int = 0
    neg_count: int = 0

    for batch in dataloader:
        _, _, labels = batch        
        pos_count += (labels == 1).sum().item()
        neg_count += (labels == 0).sum().item()

    return pos_count / (neg_count + pos_count)


def log_confusion_matrix(module: LightningModule, confusion_matrix: ConfusionMatrix, dataloader_idx: int, stage: str):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return

    conf_matrix = confusion_matrix.compute()

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    module.log(
        metric_util.build_metric_log_key('true_negatives', dataloader_idx, stage),
        conf_matrix[0, 0].item(),
        sync_dist=True
    )
    module.log(
        metric_util.build_metric_log_key('false_positives',dataloader_idx, stage),
        conf_matrix[0, 1].item(),
        sync_dist=True
    )
    module.log(
        metric_util.build_metric_log_key('false_negatives', dataloader_idx, stage),
        conf_matrix[1, 0].item(),
        sync_dist=True
    )
    module.log(
        metric_util.build_metric_log_key('true_positives', dataloader_idx, stage),
        conf_matrix[1, 1].item(),
        sync_dist=True
    )
    
    cm_filename = os.path.join(log_dir, f'{metric_util.build_metric_log_key('confusion_matrix', dataloader_idx, stage)}.png')
    fig.savefig(cm_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

def log_precision_recall_curve(module: LightningModule, precision_recall_curve: PrecisionRecallCurve, dataloader_idx: int, stage: str):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = precision_recall_curve.plot(score=True)
    ax.set_ylim(0, 1)

    dataset_balance: float = get_dataset_balance(module, dataloader_idx, stage)
    ax.plot([0, 1], [dataset_balance, dataset_balance], linestyle='--', color='red', label='Random Classifier')
    ax.legend()

    pr_curve_filename = os.path.join(log_dir, f'{metric_util.build_metric_log_key('precision_recall_curve', dataloader_idx, stage)}.png')
    fig.savefig(pr_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

    precision, recall, thresholds = precision_recall_curve.compute()
    precision, recall, thresholds = precision.cpu().numpy(), recall.cpu().numpy(), thresholds.cpu().numpy()
    
    data = []
    for i, threshold in enumerate(thresholds):
        data.append({
            "Threshold": threshold.item(),
            "Precision": precision[i].item(),
            "Recall": recall[i].item()
        })

    df = pd.DataFrame(data)
    df.to_csv(f'{log_dir}/{metric_util.build_metric_log_key('precision_recall_curve_data', dataloader_idx, stage)}.csv', index=False)

def log_roc_curve(module: LightningModule, roc_curve: ROC, dataloader_idx: int, stage: str):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return
    
    fig, ax = roc_curve.plot(score=True)
    ax.set_ylim(0, 1)

    ax.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')

    roc_curve_filename = os.path.join(log_dir, f'{metric_util.build_metric_log_key('roc_curve', dataloader_idx, stage)}.png')
    fig.savefig(roc_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

    fpr, tpr, thresholds = roc_curve.compute()
    fpr, tpr, thresholds = fpr.cpu().numpy(), tpr.cpu().numpy(), thresholds.cpu().numpy()
    
    data = []
    for i, threshold in enumerate(thresholds):
        data.append({
            "Threshold": threshold.item(),
            "FPR": fpr[i].item(),
            "TPR": tpr[i].item()
        })

    df = pd.DataFrame(data)
    df.to_csv(f'{log_dir}/{metric_util.build_metric_log_key('roc_curve_data', dataloader_idx, stage)}.csv', index=False)