import lightning.pytorch as L
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule
from torchmetrics import (
    ConfusionMatrix,
    PrecisionRecallCurve,
    ROC,
)
from sklearn.metrics import auc
import ppi_zoo.utils.metrics as metric_util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
    ax.set_title(metric_util.build_metric_title('Confusion Matrix', dataloader_idx, stage))

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
    
    precision, recall, thresholds = precision_recall_curve.compute()
    precision, recall, thresholds = precision.cpu().numpy(), recall.cpu().numpy(), thresholds.cpu().numpy()
    
    auc_value = auc(recall, precision)
    module.log(
        metric_util.build_metric_log_key('auprc', dataloader_idx, stage),
        auc_value,
        sync_dist=True
    )

    fig, ax = plt.subplots() #precision_recall_curve.plot(score=True)
    ax.plot(recall, precision, label=f"{module.name} (AUC = {auc_value:.3f})")

    ax.set_ylim(0, 1)
    ax.set_title(metric_util.build_metric_title('Precision Recall', dataloader_idx, stage))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    dataset_balance: float = get_dataset_balance(module, dataloader_idx, stage)
    ax.plot([0, 1], [dataset_balance, dataset_balance], linestyle='--', color='gray', label='Random Classifier')
    
    # Customize legend
    ax.legend(loc='best', frameon=False)

    # Set grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()

    pr_curve_filename = os.path.join(log_dir, f'{metric_util.build_metric_log_key('precision_recall_curve', dataloader_idx, stage)}.png')
    fig.savefig(pr_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues
    
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
    
    fpr, tpr, thresholds = roc_curve.compute()
    fpr, tpr, thresholds = fpr.cpu().numpy(), tpr.cpu().numpy(), thresholds.cpu().numpy()

    auc_value = np.trapz(tpr, fpr)
        
    fig, ax = plt.subplots()#roc_curve.plot(score=True)
    ax.plot(fpr, tpr, label=f"{module.name} (AUC = {auc_value:.3f})")

    ax.set_ylim(0, 1)
    ax.set_title(metric_util.build_metric_title('ROC', dataloader_idx, stage))
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    
    # Customize legend
    ax.legend(loc='best', frameon=False)

    # Set grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()

    roc_curve_filename = os.path.join(log_dir, f'{metric_util.build_metric_log_key('roc_curve', dataloader_idx, stage)}.png')
    fig.savefig(roc_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

    data = []
    for i, threshold in enumerate(thresholds):
        data.append({
            "Threshold": threshold.item(),
            "FPR": fpr[i].item(),
            "TPR": tpr[i].item()
        })

    df = pd.DataFrame(data)
    df.to_csv(f'{log_dir}/{metric_util.build_metric_log_key('roc_curve_data', dataloader_idx, stage)}.csv', index=False)