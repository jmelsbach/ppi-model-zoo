import pytorch_lightning as pl
from torchmetrics import (AUROC, F1Score, ROC, Precision, PrecisionRecallCurve,
                          Recall, ConfusionMatrix)

class AbstractPPIClassifierModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()

        # define measures
        self._auroc = AUROC(pos_label=1)
        self._f1 = F1Score(num_classes=2, average='macro')
        self._precision = Precision(num_classes=2, average='macro')
        self._recall = Recall(num_classes=2, average='macro')
        self._confusion_matrix = ConfusionMatrix(num_classes=2)

        # final results stored in the model
        self.test_results = {}

    # temp: adjust the features needed to perform a prediction accordingly
    def __call__(self, prot1, prot2):
        raise NotImplementedError("Call method must be implemented by the subclass")
    
    # temp: adjust the features needed to perform a prediction and potential functionality accordingly
    def forward(self, prot1, prot2):
        raise NotImplementedError("Forward method must be implemented by the subclass")
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step must be implemented by the subclass")

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer configuration must be implemented by the subclass")

    def test_step(self, batch, batch_idx):
        # temp: adjust the features needed to perform a prediction accordingly
        prot1, prot2, labels = batch
        predictions = self(prot1, prot2)
        
        self._auroc.update(predictions, labels)
        self._f1.update(predictions, labels)
        self._precision.update(predictions, labels)
        self._recall.update(predictions, labels)
        self._confusion_matrix.update(predictions, labels)

        return {'outputs': predictions, 'labels': labels}

    def test_epoch_end(self, outputs):
        # Compute final metric values
        self.test_results['auroc'] = self._auroc.compute()
        self.test_results['f1'] = self._f1.compute()
        self.test_results['precision'] = self._precision.compute()
        self.test_results['recall'] = self._recall.compute()
        self.test_results['confusion_matrix'] = self._confusion_matrix.compute()
        
        # Log metrics
        self.log('test_auroc', self.test_results['auroc'])
        self.log('test_f1', self.test_results['f1'])
        self.log('test_precision', self.test_results['precision'])
        self.log('test_recall', self.test_results['recall'])
        self.log('test_confusion_matrix', self.test_results['confusion_matrix'])
        
        # Reset metrics
        self._auroc.reset()
        self._f1.reset()
        self._precision.reset()
        self._recall.reset()
        self._confusion_matrix.reset()
        
        return self.test_results