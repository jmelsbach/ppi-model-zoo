import torch
import lightning.pytorch as L
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel, AutoTokenizer, BertConfig
from collections import OrderedDict
from typing import List

# TODO: add logging  and wandb Trainer(logger = wandb_logger)!
# [TODO] add standard parameters from STEP Paper
# TODO: metrics -> after merge // Dependency injection (Johannes: not include metrics in model but in constructor -> low prio)
# TODO: scheduling? -> Done, but test if it actually works
# TODO: label encoder? -> low prio
# TODO: predict methods -> low prio
# TODO: hyperparameter welcher steuert ob man Adam oder AdamW verwendet -> low prio

# Questions:
# - does scheduling work? -> log with wandb (use lightning https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html)
# - local_logger?
# - global_rank == 0?
# - keine activation function im classification head? -> CHECK WITH STEP PAPER


class STEP(L.LightningModule):
    # TODO: standardwerte die den paper entsprechen
    def __init__(
        self,
        learning_rate: float = 0.001,
        nr_frozen_epochs: int = 2,
        dropout_rates: List[float] = [0.1, 0.2, 0.2],
        encoder_features: int = 1024,
        model_name: str = 'Rostlab/prot_bert_bfd',
        pool_cls: bool = True,
        pool_max: bool = True,
        pool_mean: bool = True,
        pool_mean_sqrt: bool = True,
        weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        warumup_steps: int = 200,
        encoder_learning_rate: float = 5e-06
    ) -> None:
        """
        Siamese Tailored deep sequence Embedding of Proteins (STEP) model
        Reference: https://github.com/SCAI-BIO/STEP

        Args:
            learning_rate: learning rate for the optimizer
            nr_frozen_epochs: number of epochs the encoder should be frozen
            dropout_rates: dropout probability
            encoder_features: number of features the encoder outputs
            model_name: name of the pretrained model
            pool_cls: Applies pooling over the CLS token (representing the whole amino acid sequence) // --> Use to determine input and output dimensions of the model
            pool_max: Applies max pooling over the token embeddings, considering only valid tokens. // --> Use to determine input and output dimensions of the model
            pool_mean: Computes the mean of the token embeddings, considering only valid tokens. // --> Use to determine input and output dimensions of the model
            pool_mean_sqrt: // --> Use to determine input and output dimensions of the model
            label_set: WIP
            max_length: WIP
            warmup_steps: WIP
            encoder_learning_rate: WIP
            weight_decay: WIP
            adam_epsilon: WIP
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.nr_frozen_epochs = nr_frozen_epochs
        self.dropout_rates = dropout_rates
        self.encoder_features = encoder_features
        self.model_name = model_name
        # number of features the encoder outputs
        self.encoder_features = encoder_features
        self.pool_cls = pool_cls
        self.pool_max = pool_max
        self.pool_mean = pool_mean
        self.pool_mean_sqrt = pool_mean_sqrt
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warumup_steps
        self.encoder_learning_rate = encoder_learning_rate

        self.save_hyperparameters(ignore=[])

        self._build_model()

        if self.nr_frozen_epochs > 0:
            self._freeze_encoder()
        else:
            self._frozen = False

        self.loss_function = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        train_loss = self._single_step(batch)
        self.log('train_loss', train_loss)
        self.log('frozen', self._frozen)

        return train_loss

    def on_train_epoch_end(self) -> None:
        # TODO: STEP logs the training metrics here
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

    def configure_optimizers(self) -> tuple:
        """
        Confiugre the optimizers and schedulers.

        It also sets different learning rates for different parameter groups. 
        """
        no_decay_params = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [param for name, param in self.ProtBertBFD.named_parameters() if not any(ndp in name for ndp in no_decay_params)],
                "lr": self.encoder_learning_rate,
            },
            {
                "params": [param for name, param in self.ProtBertBFD.named_parameters() if any(ndp in name for ndp in no_decay_params)],
                "weight_decay": 0.0,
                "lr": self.encoder_learning_rate,
            },
            {
                "params": self.classification_head.parameters(),
            },
        ]

        parameters = optimizer_grouped_parameters
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
            # betas = self.hparams.betas
        )

        scheduler = LambdaLR(optimizer, self._lr_lambda)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
            'monitor': 'val_loss',
            'name': 'learning_rate'
        }

        return [optimizer], [scheduler_dict] # [TODO] return dict with optimizer and scheduler

    def forward(self, inputs_A, inputs_B) -> torch.Tensor:
        # = torch.Size([8, 8, 1024]) -> torch.Size([num_sequences,  num_tokens_per_sequence, embedding_vectors_for_token])
        token_embeddings_A = self._compute_embedding(inputs_A)
        token_embeddings_B = self._compute_embedding(inputs_B)

        # attention_mask is a tensor that indicates which tokens are valid (not padding) with 1s and which are padding with 0s.
        attention_mask_A = inputs_A['attention_mask']
        attention_mask_B = inputs_B['attention_mask']

        # concatenate embeddings after applying pooling strategy (_pooling())
        embeddings_pooled_A = self._pooling(
            token_embeddings_A, attention_mask_A
        )
        embeddings_pooled_B = self._pooling(
            token_embeddings_B, attention_mask_B
        )

        x = embeddings_pooled_A * embeddings_pooled_B
        classifier_output = self.classification_head(x)
        # reshaping output vector to 1D tensor
        return classifier_output.view(-1)

    def _build_model(self) -> None:
        # Build ProtBert Encoder
        config = BertConfig.from_pretrained(self.model_name)
        config.gradient_checkpointing = True
        # pre-trained model configuration, gradient_checkpoints, label_encoder and other configs??
        # One important difference between our Bert model and the original Bert version is the way of dealing with sequences as separate documents
        self.ProtBertBFD = AutoModel.from_pretrained(
            self.model_name,
            config=config
        )
        # This means the Next sentence prediction is not used, as each sequence is treated as a complete document.
        # The masking follows the original Bert training with randomly masks 15% of the amino acids in the input.

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, do_lower_case=False)

        # Build classification head
        self.total_encoder_features = sum([
            self.encoder_features if self.pool_cls else 0,
            self.encoder_features if self.pool_max else 0,
            self.encoder_features if self.pool_mean else 0,
            self.encoder_features if self.pool_mean_sqrt else 0
        ])

        self.classification_head = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(self.dropout_rates[0])),
            ("dense1", nn.Linear(self.total_encoder_features,
             int(self.total_encoder_features / 16))),
            ("dropout2", nn.Dropout(self.dropout_rates[1])),
            ("dense2", nn.Linear(int(self.total_encoder_features / 16),
             int(self.total_encoder_features / (16*16)))),
            ("dropout3", nn.Dropout(self.dropout_rates[2])),
            ("dense3", nn.Linear(int(self.total_encoder_features / (16*16)), 1)),
        ]))

    def _compute_embedding(self, inputs) -> torch.Tensor:
        embeddings = self.ProtBertBFD(  # embeddings.shape = torch.Size([8, 8, 1024])
            inputs['input_ids'], inputs['attention_mask'])[0]  # returns the last_hidden_state of the model whereby [1] would return the pooler_output
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

    def _pooling(self, token_embeddings, attention_mask, pool_cls=True, pool_max=True, pool_mean=True, pool_mean_sqrt=True) -> torch.Tensor:
        """
        pool_max: Applies max pooling over the token embeddings, considering only valid tokens.
            pool_mean: Computes the mean of the token embeddings, considering only valid tokens.
                pool_mean_sqrt: Computes the mean of the token embeddings, adjusted by the square root of the number of valid tokens.
        """

        # extract the embedding vector of the first token (=CLS token) which capture contextual information from the entire sequence.
        cls_token_embeddings = token_embeddings[:, 0]

        # Pooling strategy
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        # -> token_id = 2 -> Protbert(token_id) -> embedding_vector_of_token_cls (1024)
        if pool_cls:
            output_vectors.append(cls_token_embeddings)
        if pool_max:
            # attention_mask is a tensor that indicates which tokens are valid (not padding) with 1s and which are padding with 0s.
            # unsqueeze(-1) adds a new dimension at the end, making it compatible for broadcasting.
            # expand(token_embeddings.size()) expands the mask to match the size of token_embeddings.
            # The result is a mask with the same shape as token_embeddings but with 1s for valid tokens and 0s for padding tokens.
            # Set padding tokens to large negative value
            token_embeddings[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded,
                1
            )

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            # if 'token_weights_sum' in token_embeddings:
            #     sum_mask = token_embeddings['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            # else:
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        # shape -> torch.Size([8, 4096]) -> torch.Size([num_sequences,  3 (pool_cls, pool_max, pool_mean) * embedding_vectors_for_token])
        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def _lr_lambda(self, current_step: int) -> float:
        """
        Calculate learning rate for current step according to the total number of training steps

        Args:
            current_step (int): Current step number

        Returns:
            [float]: learning rate lambda (how much the rate should be changed.)
        """
        num_warmup_steps = self.warmup_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) /
            float(max(1, self.num_training_steps - num_warmup_steps))
        )

    @property
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps
