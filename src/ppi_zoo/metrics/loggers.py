from lightning.pytorch import LightningModule
from torchmetrics import (
    ConfusionMatrix,
    PrecisionRecallCurve,
    ROC,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def log_confusion_matrix(module: LightningModule, confusion_matrix: ConfusionMatrix, dataloader_idx: int):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return

    conf_matrix = confusion_matrix.compute()

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    module.log(f'true_negatives_{dataloader_idx}', conf_matrix[0, 0].item(), sync_dist=True)
    module.log(f'false_positives_{dataloader_idx}', conf_matrix[0, 1].item(), sync_dist=True)
    module.log(f'false_negatives_{dataloader_idx}', conf_matrix[1, 0].item(), sync_dist=True)
    module.log(f'true_positives_{dataloader_idx}', conf_matrix[1, 1].item(), sync_dist=True)
    
    cm_filename = os.path.join(log_dir, f'confusion_matrix_{dataloader_idx}.png')
    fig.savefig(cm_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

def log_precision_recall_curve(module: LightningModule, precision_recall_curve: PrecisionRecallCurve, dataloader_idx: int):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return
    
    fig, ax = precision_recall_curve.plot(score=True)
    
    pr_curve_filename = os.path.join(log_dir, f'precision_recall_curve_{dataloader_idx}.png')
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
    df.to_csv(f'{log_dir}/precision_recall_curve_data_{dataloader_idx}.csv', index=False)

def log_roc_curve(module: LightningModule, roc_curve: ROC, dataloader_idx: int):
    log_dir = module.logger.log_dir
    if log_dir is None:
        return
    
    fig, ax = roc_curve.plot(score=True)
    
    roc_curve_filename = os.path.join(log_dir, f'roc_curve_{dataloader_idx}.png')
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
    df.to_csv(f'{log_dir}/roc_curve_data_{dataloader_idx}.csv', index=False)