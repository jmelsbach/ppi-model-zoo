import pandas as pd
from src.datasets.GoldStandardData import GoldStandardDataModule

#with open('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl', 'rb') as file:
#    data = pkl.load(file)

#data.to_csv('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv', index=False)
#df = pd.read_csv('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv')

"""
datamodule = GoldStandardDataModule('../../.data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv')
datamodule.setup()
traindataloader = datamodule.train_dataloader()
first_batch = next(iter(traindataloader))
print(first_batch)
"""
print('Hello')