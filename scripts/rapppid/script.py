from ppi_zoo.datasets.RapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging

data_module = RapppidDataModule(data_dir = '.data', file_name = 'benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv', batch_size = 80, tokenizer_file = 'scripts/rapppid/spm.model', with_validation = True, truncate_len = 1500)
# todo: check swa parameter ()Enable Stochastic Weight Averaging.) which should be true
model = LSTMAWD(num_codes=250, embedding_size=64, steps_per_epoch=None, num_epochs=25, lstm_dropout_rate=0.3, # todo: rename num_codes? (num_codes is the vocab_size of the tokenizer); do we num_epochs?
                classhead_dropout_rate=0.2, rnn_num_layers=2, classhead_num_layers=2, lr=0.01, # todo: modified lr
                weight_decay=0.0001, bi_reduce='last', class_head_name='mult', variational_dropout=False,
                lr_scaling=False, trunc_len=1500, embedding_droprate=0.3, frozen_epochs=0, optimizer_type='adam') # todo: d
# todo: test StochasticWeightAveraging
trainer = Trainer(accelerator="gpu", devices=1, max_epochs=25, callbacks=[StochasticWeightAveraging(0.05)]) #, accelerator='gpu', devices=1)
trainer.fit(model=model, datamodule=data_module)

# num_epochs = 100 -> set checkpoint automatically to lowest validation loss