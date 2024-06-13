import pytorch_lightning as L
from ppi_zoo.datasets.GoldStandardData import GoldStandardDataModule
from ppi_zoo.models.step.model import STEP
from transformers import AutoTokenizer

step = STEP(
    learning_rate=0.001,
    nr_frozen_epochs=2
)

# TODO: callbacks angucken (vlt. MetricsCallback)

datamodule = GoldStandardDataModule(
    data_dir='../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv',
    batch_size=8,
    tokenizer=AutoTokenizer.from_pretrained(
        "Rostlab/prot_bert_bfd", do_lower_case=False),
)

trainer = L.Trainer(max_epochs=5)
trainer.fit(step, datamodule)
trainer.test(step, datamodule)
