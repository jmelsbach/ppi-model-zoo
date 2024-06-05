import torch
import pytorch_lightning as pl
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer
from collections import OrderedDict

class STEP(pl.LightningModule):
    def __init__(self, learning_rate:float=0.001):
        super().__init__()

        encoder_features = 1024
        model_name = 'Rostlab/prot_bert_bfd'
        config = BertConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True
        
        self.ProtBertBFD = BertModel.from_pretrained(model_name, config=config)
        self.ProtBertBFD.output = torch.nn.Linear(4096, 2)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

        self.classification_head = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(encoder_features, int(encoder_features * 4 / 16))),
            ('l2', nn.Linear(int(encoder_features * 4 / 16), 1))
        ]))

        self.loss_function = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, input_ids, token_type_ids, attention_mask):
        word_embeddings = self.ProtBertBFD(input_ids, attention_mask)[0]
        cls_token_embeddings = word_embeddings[:, 0]

        output_vectors = []
        output_vectors.append(cls_token_embeddings)
        output_vectors = torch.cat(output_vectors, 1)

        return output_vectors
    
    def _single_step(self, batch):
        inputs_A, inputs_B, targets = batch
        pred_A = self.forward(**inputs_A)
        pred_B = self.forward(**inputs_B)

        x = pred_A * pred_B
        classifier_output = self.classification_head(x)
        classifier_output = classifier_output.view(-1)

        return self.loss_function(classifier_output, targets.float())

    def training_step(self, batch, batch_idx):
        train_loss = self._single_step(batch)
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._single_step(batch)
        self.log('val_loss', val_loss)

        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self._single_step(batch)
        self.log('test_loss', test_loss)

        return test_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
