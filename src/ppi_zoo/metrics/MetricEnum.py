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
    'accuracy': { 'metric': Accuracy(task='binary') },
    'f1': { 'metric': F1Score(task='binary') },
    'recall': { 'metric': Recall(task='binary') },
    'precision': { 'metric': Precision(task='binary') },
    'auroc': { 'metric': AUROC(task='binary') },
    'confusion_matrix': { 
        'metric': ConfusionMatrix(task='binary'),
        'log': loggers.log_confusion_matrix 
    },
    'precision_recall_curve': {
        'metric': PrecisionRecallCurve(task='binary'),
        'log': loggers.log_precision_recall_curve
    },
    'roc_curve': {
        'metric': ROC(task='binary'),
        'log': loggers.log_roc_curve
    }
}