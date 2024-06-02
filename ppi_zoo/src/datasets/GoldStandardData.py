from torch.utils.data import Dataset
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class GoldStandardDataset(Dataset):

    def __init__(self, data_dir:str='.data/'):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        return item

class GoldStandardDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str='.data', batch_size:int= 32, num_workers:int= 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        dataset = GoldStandardDataset(data_dir=self.data_dir)
        
        train_indices = dataset.data.index[dataset.data['trainTest'] == 'train'].tolist()
        val_indices = dataset.data.index[dataset.data['trainTest'] == 'test1'].tolist()
        test_indices = dataset.data.index[dataset.data['trainTest'] == 'test2'].tolist()

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)