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
model = LSTMAWD()

trainer = Trainer(accelerator="gpu", devices=1, max_epochs=25, callbacks=[StochasticWeightAveraging(0.05)])#, accelerator='gpu', devices=1)
trainer.fit(model=model, datamodule=data_module)