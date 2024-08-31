from ppi_zoo.datasets.RapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging

data_module = RapppidDataModule(
    data_dir = '.data',
    file_name = 'benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv',
    batch_size = 80,
    tokenizer_file = 'scripts/rapppid/spm.model',
    with_validation = True,
    truncate_len = 1500
)

# todo: check swa parameter ()Enable Stochastic Weight Averaging.) which should be true
model = LSTMAWD()
# todo: test StochasticWeightAveraging
trainer = Trainer(accelerator="gpu", devices=1, max_epochs=25, callbacks=[StochasticWeightAveraging(0.05)]) #, accelerator='gpu', devices=1)
trainer.fit(model=model, datamodule=data_module)

# num_epochs = 100 -> set checkpoint automatically to lowest validation loss