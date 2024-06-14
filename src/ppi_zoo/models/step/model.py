import torch
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict
from typing import List

# TODO: scheduling?
# TODO: label encoder?
# TODO: predict methods
# TODO: logging
# auf jeden Fall benutzen
# learning rate finder bestimmt max für scheduler

class STEP(pl.LightningModule):
    # TODO: explizite parameter, standardwerte die den paper entsprechen
    def __init__(self, learning_rate: float = 0.001, nr_frozen_epochs: int = 0, dropout_rate: List[float] = [0.1, 0.2, 0.2],
                 encoder_features: int = 1024, model_name: str = 'Rostlab/prot_bert_bfd', **config):
        """
        Possible hyperparameters:
        - learning_rate (float): learning rate for the optimizer
        - nr_frozen_epochs (int): number of epochs the encoder should be frozen
        - dropout_rate (float): dropout probability
        - encoder_features (int): number of features the encoder outputs
        - model_name: name of the pretrained model
        - config (dict): Additional configuration parameters:
            - pool_cls: Applies pooling over the CLS token (representing the whole amino acid sequence) // --> Use to determine input and output dimensions of the model
            - pool_max: Applies max pooling over the token embeddings, considering only valid tokens. // --> Use to determine input and output dimensions of the model
            - pool_mean: Computes the mean of the token embeddings, considering only valid tokens. // --> Use to determine input and output dimensions of the model
            - pool_mean_sqrt // --> Use to determine input and output dimensions of the model
            - label_set?
            - max_length?
            - warmup_steps?
            - encoder_learning_rate?
            - weight_decay?
            - adam_epsilon?
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.nr_frozen_epochs = nr_frozen_epochs
        self.dropout_rate = dropout_rate
        self.encoder_features = encoder_features
        self.model_name = model_name
        self.encoder_features = encoder_features

        # pre-trained model configuration, gradient_checkpoints, label_encoder and other configs??
        self.ProtBertBFD = AutoModel.from_pretrained(model_name)    # One important difference between our Bert model and the original Bert version is the way of dealing with sequences as separate documents 
                                                                    # This means the Next sentence prediction is not used, as each sequence is treated as a complete document. 
                                                                    # The masking follows the original Bert training with randomly masks 15% of the amino acids in the input.
                                                        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=False)

        if self.nr_frozen_epochs > 0:
            self._freeze_encoder()
        else:
            self._frozen = False
        
        # number of features the encoder outputs
        self.total_encoder_features = 0
        if config['pool_cls']:
            self.total_encoder_features += encoder_features
        if config['pool_max']:
            self.total_encoder_features += encoder_features
        if config['pool_mean']:
            self.total_encoder_features += encoder_features
        if config['pool_mean_sqrt']:
            self.total_encoder_features += encoder_features        
        # classification head --> [TODO] NO ACTIVATION FUNCTIONS (check with paper!!!)
        self.classification_head = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(self.dropout_rate[0])), 
            ("dense1", nn.Linear(self.total_encoder_features, int(self.total_encoder_features / 16))),
            ("dropout2", nn.Dropout(self.dropout_rate[1])),
            ("dense2", nn.Linear(int(self.total_encoder_features / 16),
             int(self.total_encoder_features / (16*16)))),
            ("dropout3", nn.Dropout(self.dropout_rate[2])),
            ("dense3", nn.Linear(int(self.total_encoder_features / (16*16)), 1)),
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

    def forward(self, inputs_A, inputs_B) -> torch.Tensor: # 
        token_embeddings_A = self._compute_embedding(inputs_A) # = torch.Size([8, 8, 1024]) -> torch.Size([num_sequences,  num_tokens_per_sequence, embedding_vectors_for_token])
        token_embeddings_B = self._compute_embedding(inputs_B)

        attention_mask_A = inputs_A['attention_mask'] # attention_mask is a tensor that indicates which tokens are valid (not padding) with 1s and which are padding with 0s.
        attention_mask_B = inputs_B['attention_mask']

        # concatenate embeddings after applying pooling strategy (_pooling())
        embeddings_pooled_A = self._pooling(token_embeddings_A, attention_mask_A)
        embeddings_pooled_B = self._pooling(token_embeddings_B, attention_mask_B)

        x = embeddings_pooled_A * embeddings_pooled_B
        classifier_output = self.classification_head(x)
        return classifier_output.view(-1) # reshaping output vector to 1D tensor

    def _compute_embedding(self, inputs) -> torch.Tensor:
        embeddings = self.ProtBertBFD( # embeddings.shape = torch.Size([8, 8, 1024])
            inputs['input_ids'], inputs['attention_mask'])[0] # returns the last_hidden_state of the model whereby [1] would return the pooler_output 
        return embeddings
           # inputs['input_ids'] = tokenized sequence with shape torch.Size([8, 8]) -> 8 sequences with 8 tokens each
        # tensor([[2, 1, 3, 0, 0, 0, 0, 0], note: 2 is the start token, 1 is the sequence token, 3 is the end token, 0 is the padding token
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0]], device='cuda:0')

        # inputs['attention_mask'] = tensor([[2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0],
        # [2, 1, 3, 0, 0, 0, 0, 0]], device='cuda:0')}

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

    def _pooling(self, token_embeddings, attention_mask, pool_cls=True, pool_max=True, pool_mean=True, pool_mean_sqrt=True):
        """
        pool_max: Applies max pooling over the token embeddings, considering only valid tokens.
	    pool_mean: Computes the mean of the token embeddings, considering only valid tokens.
		pool_mean_sqrt: Computes the mean of the token embeddings, adjusted by the square root of the number of valid tokens.
        """
        
        cls_token_embeddings = token_embeddings[:, 0] # extract the embedding vector of the first token (=CLS token) which capture contextual information from the entire sequence.

        ## Pooling strategy
        output_vectors = []
        if pool_cls: # -> token_id = 2 -> Protbert(token_id) -> embedding_vector_of_token_cls (1024)
            output_vectors.append(cls_token_embeddings)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # attention_mask is a tensor that indicates which tokens are valid (not padding) with 1s and which are padding with 0s.
            # unsqueeze(-1) adds a new dimension at the end, making it compatible for broadcasting.
	        # expand(token_embeddings.size()) expands the mask to match the size of token_embeddings.
	        # The result is a mask with the same shape as token_embeddings but with 1s for valid tokens and 0s for padding tokens.
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            # if 'token_weights_sum' in token_embeddings:
            #     sum_mask = token_embeddings['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            # else:
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1) # shape -> torch.Size([8, 4096]) -> torch.Size([num_sequences,  3 (pool_cls, pool_max, pool_mean) * embedding_vectors_for_token])
        return output_vector 