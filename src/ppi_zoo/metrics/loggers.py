from lightning.pytorch.loggers import Logger
from torchmetrics import (
    ConfusionMatrix,
    PrecisionRecallCurve,
    ROC,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def log_confusion_matrix(logger: Logger, confusion_matrix: ConfusionMatrix, dataloader_idx: int):
    conf_matrix = confusion_matrix.compute()
    conf_matrix_dict = {
        f'true_negatives_{dataloader_idx}': conf_matrix[0, 0].item(),
        f'false_positives_{dataloader_idx}': conf_matrix[0, 1].item(),
        f'false_negatives_{dataloader_idx}': conf_matrix[1, 0].item(),
        f'true_positives_{dataloader_idx}': conf_matrix[1, 1].item()
    }
    #logger.log_dict(conf_matrix_dict)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    # Save plot to file
    log_dir = logger.log_dir
    if log_dir is None:
        plt.close(fig)
        return
    
    cm_filename = os.path.join(log_dir, f'confusion_matrix_{dataloader_idx}.png')
    fig.savefig(cm_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

def log_precision_recall_curve(logger, precision_recall_curve: PrecisionRecallCurve, dataloader_idx: int):
    fig, ax = precision_recall_curve.plot(score=True)
    # Save plot to file
    log_dir = logger.log_dir
    if log_dir is None:
        plt.close(fig)  # Close the figure to avoid memory issues
        return
    
    pr_curve_filename = os.path.join(log_dir, f'precision_recall_curve_{dataloader_idx}.png')
    fig.savefig(pr_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues

def log_roc_curve(logger, roc_curve: ROC, dataloader_idx: int):
    fig, ax = roc_curve.plot(score=True)
    # Save plot to file
    log_dir = logger.log_dir
    if log_dir is None:
        plt.close(fig)  # Close the figure to avoid memory issues
        return
    
    roc_curve_filename = os.path.join(log_dir, f'roc_curve_{dataloader_idx}.png')
    fig.savefig(roc_curve_filename)
    plt.close(fig)  # Close the figure to avoid memory issues
