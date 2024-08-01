from ppi_zoo.datasets.RapppidDataset import RapppidDataModule

data_module = RapppidDataModule(data_dir = '/home/mwlcek-asim/repos/ppi-model-zoo/.data', file_name = 'benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.csv', tokenizer_file = '/home/mwlcek-asim/repos/ppi-model-zoo/scripts/rapppid/spm.model', with_validation = True, truncate_len = 1000)
data_module.setup()
train_dl = data_module.train_dataloader()
for x in train_dl:
    print(x)
    break