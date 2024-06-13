import torch
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict

# TODO: pool strategy
# TODO: label encoder?
# TODO: predict methods
# TODO: logging
# TODO: scheduling?
# auf jeden Fall benutzen
# learning rate finder bestimmt max für scheduler

class STEP(pl.LightningModule):
    # TODO: explizite parameter, standardwerte die den paper entsprechen
    def __init__(self, learning_rate: float = 0.001, nr_frozen_epochs: int = 0):
        super().__init__()

        encoder_features = 1024
        model_name = 'Rostlab/prot_bert_bfd'

        self.ProtBertBFD = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False)
        self._frozen = False
        self.nr_frozen_epochs = nr_frozen_epochs

        if self.nr_frozen_epochs > 0:
            self._freeze_encoder()

        self.classification_head = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(0.1)), # TODO: dropout rate hyperparameter
            ("dense1", nn.Linear(encoder_features, int(encoder_features / 16))),
            ("dropout2", nn.Dropout(0.2)),
            ("dense2", nn.Linear(int(encoder_features / 16),
             int(encoder_features / (16*16)))),
            ("dropout3", nn.Dropout(0.2)),
            ("dense3", nn.Linear(int(encoder_features / (16*16)), 1)),
        ]))

        self.loss_function = nn.BCEWithLogitsLoss()  # vlt auch CrossEntropyLoss
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        train_loss = self._single_step(batch)
        self.log('train_loss', train_loss)

        return train_loss

    def on_train_epoch_end(self) -> None:
        if self.current_epoch + 1 > self.nr_frozen_epochs:
            self._unfreeze_encoder()

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        val_loss = self._single_step(batch)
        self.log('val_loss', val_loss)

        return val_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        test_loss = self._single_step(batch)
        self.log(f'test_loss', test_loss)

        return test_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # TODO: Adam vs AdamW?
        # TODO: hyperparameter welcher steuert ob man Adam oder AdamW verwendet
        # TODO: weight decay und epsilon
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs_A, inputs_B) -> torch.Tensor:
        embeddings_A = self._compute_embedding(inputs_A)
        embeddings_B = self._compute_embedding(inputs_B)

        x = embeddings_A * embeddings_B
        classifier_output = self.classification_head(x)
        return classifier_output.view(-1)

    def _compute_embedding(self, inputs) -> torch.Tensor:
        embeddings = self.ProtBertBFD(
            inputs['input_ids'], inputs['attention_mask'])[0]
        return embeddings[:, 0]

    def _single_step(self, batch) -> torch.Tensor:
        inputs_A, inputs_B, targets = batch
        return self.loss_function(
            self.forward(inputs_A, inputs_B),
            targets.float()
        )

    def _freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.ProtBertBFD.parameters():
            param.requires_grad = False
        self._frozen = True

    def _unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if not self._frozen:
            return

        for param in self.ProtBertBFD.parameters():
            param.requires_grad = True
        self._frozen = False
