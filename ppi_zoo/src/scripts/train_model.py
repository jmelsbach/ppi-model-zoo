import pytorch_lightning as L

from ppi_zoo.src.datasets.GoldStandardData import GoldStandardDataModule
from transformers import BertTokenizer
from ppi_zoo.src.models.step.model import STEP

step = STEP(learning_rate=0.001)

datamodule = GoldStandardDataModule(
    data_dir='../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv',
    batch_size=8, 
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
)

trainer = L.Trainer()
trainer.fit(step, datamodule)
