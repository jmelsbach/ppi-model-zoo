import pytorch_lightning as L
from ppi_zoo.datasets.GoldStandardData import GoldStandardDataModule
from ppi_zoo.models.step.model import STEP
from transformers import AutoTokenizer

# Model initialization
step = STEP(
    learning_rate=0.001,
    nr_frozen_epochs=2,
    dropout_rate=[0.1, 0.2, 0.2],
    encoder_features=1024,
    model_name = 'Rostlab/prot_bert_bfd',
    pool_cls=True,
    pool_max=True,
    pool_mean=True,
    pool_mean_sqrt=True
)

# TODO: callbacks angucken (vlt. MetricsCallback)

datamodule = GoldStandardDataModule(
    data_dir='.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv',
    batch_size=8,
    tokenizer=AutoTokenizer.from_pretrained(
        "Rostlab/prot_bert_bfd", do_lower_case=False),
    max_len=8, # currently set to 8 because of cuda out of memory error
    )

trainer = L.Trainer(max_epochs=5)
trainer.fit(step, datamodule)
trainer.test(step, datamodule)
