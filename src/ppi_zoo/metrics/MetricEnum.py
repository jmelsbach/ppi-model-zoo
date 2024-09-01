from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    PrecisionRecallCurve,
    F1Score,
    ROC,
    AUROC
)
import ppi_zoo.metrics.loggers as loggers

METRIC_ENUM = {
    'accuracy': { 'build_metric': lambda: Accuracy(task='binary') },
    'f1': { 'build_metric': lambda: F1Score(task='binary') },
    'weighted_f1': { 'build_metric': lambda: F1Score(task='binary', average="weighted") },
    'recall': { 'build_metric': lambda: Recall(task='binary') },
    'precision': { 'build_metric': lambda: Precision(task='binary') },
    'auroc': { 'build_metric': lambda: AUROC(task='binary') },
    'confusion_matrix': { 
        'build_metric': lambda: ConfusionMatrix(task='binary'),
        'log': loggers.log_confusion_matrix 
    },
    'precision_recall_curve': {
        'build_metric': lambda: PrecisionRecallCurve(task='binary'),
        'log': loggers.log_precision_recall_curve
    },
    'roc_curve': {
        'build_metric': lambda: ROC(task='binary'),
        'log': loggers.log_roc_curve
    }
}