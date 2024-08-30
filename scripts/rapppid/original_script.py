from ppi_zoo.datasets.OriginalRapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging

import os
import sys

def _getThreads():
        """ Returns the number of available threads on a posix/win based system """
        if sys.platform == 'win32':
            return (int)(os.environ['NUMBER_OF_PROCESSORS'])
        else:
            return (int)(os.popen('grep -c cores /proc/cpuinfo').read())

workers =  max(1, _getThreads()-2)


data_module = RapppidDataModule(batch_size = 80, train_path = '.data/comparatives/string_c3/train_pairs.pkl.gz', val_path ='.data/comparatives/string_c3/val_pairs.pkl.gz', test_path = '.data/comparatives/string_c3/test_pairs.pkl.gz', seqs_path = '.data/comparatives/string_c3/seqs.pkl.gz', 
                                trunc_len = 1500, workers = workers, vocab_size = 250, model_file='scripts/rapppid/spm.model', seed=5353456)
data_module.setup()

model = LSTMAWD(num_codes=250, embedding_size=64, steps_per_epoch=None, num_epochs=25, lstm_dropout_rate=0.3, # todo: rename num_codes? (num_codes is the vocab_size of the tokenizer); do we num_epochs?
                classhead_dropout_rate=0.2, rnn_num_layers=2, classhead_num_layers=2, lr=0.003,
                weight_decay=0.0001, bi_reduce='last', class_head_name='mult', variational_dropout=False,
                lr_scaling=False, trunc_len=1500, embedding_droprate=0.3, frozen_epochs=0, optimizer_type='adam') # todo: d

trainer = Trainer(accelerator="gpu", devices=1, max_epochs=25, callbacks=[StochasticWeightAveraging(0.05)])#, accelerator='gpu', devices=1)
trainer.fit(model=model, datamodule=data_module)