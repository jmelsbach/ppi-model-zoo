import pandas as pd
from ppi_zoo.datasets.GoldStandardData import GoldStandardDataModule
from transformers import BertModel, BertConfig, BertTokenizer

#with open('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl', 'rb') as file:
#    data = pkl.load(file)

#data.to_csv('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv', index=False)
#df = pd.read_csv('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv')

datamodule = GoldStandardDataModule(
    data_dir='../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv', 
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False),
    train_val_split=(0.9, 0.1)
)
datamodule.setup()

traindataloader = datamodule.train_dataloader()
print(len(traindataloader))
print(len(datamodule.val_dataloader()))
#first_batch = next(iter(traindataloader))
#uniprotID_Aa, uniprotID_B, isInteraction, trainTest, RNAseqHPA, tissueHPA, tissueCellHPA, subcellularLocationHPA, bioProcessUniprot, cellCompUniprot, molFuncUniprot, domainUniprot, motifUniprot, Bgee, sequence_A, sequence_B = first_batch
#print(first_batch)
#model_name = 'Rostlab/prot_bert_bfd'
#config = BertConfig.from_pretrained(model_name)
#config.gradient_checkpointing = True
#ProtBertBFD = BertModel.from_pretrained(model_name, config=config)
#print(ProtBertBFD)
