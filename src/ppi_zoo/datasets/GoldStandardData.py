from torch.utils.data import Dataset
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class GoldStandardDataset(Dataset):

    def __init__(self, data_dir:str, tokenizer:object, max_len:int):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        # TODO: label as tensor
        target = item['isInteraction']

        tokens_A = self.tokenizer(sequence_A, max_length=self.max_len, add_special_tokens=True, padding="max_length",
                            truncation=True, return_tensors='pt')
        tokens_B = self.tokenizer(sequence_B, max_length=self.max_len, add_special_tokens=True, padding="max_length",
                            truncation=True, return_tensors='pt')

        # ist das nötig?
        # shape printen gucken ob wir squeeze
        tokens_A['input_ids'] = torch.as_tensor(tokens_A['input_ids'], dtype=torch.long).squeeze()
        tokens_A['attention_mask'] = torch.as_tensor(tokens_A['attention_mask'], dtype=torch.long).squeeze()
        tokens_A['token_type_ids'] = torch.as_tensor(tokens_A['token_type_ids'], dtype=torch.long).squeeze()

        tokens_B['input_ids'] = torch.as_tensor(tokens_B['input_ids'], dtype=torch.long).squeeze()
        tokens_B['attention_mask'] = torch.as_tensor(tokens_B['attention_mask'], dtype=torch.long).squeeze()
        tokens_B['token_type_ids'] = torch.as_tensor(tokens_B['token_type_ids'], dtype=torch.long).squeeze()
        return tokens_A, tokens_B, target

class GoldStandardDataModule(pl.LightningDataModule):
    # TODO: hier url übergeben
    # url aus map rausnehmen für user
    def __init__(self, data_dir:str='.data', batch_size:int= 2, num_workers:int= 4, tokenizer: object=None, max_len:int=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # TODO: download data from cloud?
    def setup(self, stage=None):
        dataset = GoldStandardDataset(data_dir=self.data_dir, tokenizer=self.tokenizer, max_len=self.max_len)
        
        # TODO: zwei test sets test1 test2, für validation Stück von training data nehmen
        # wenn wir aus training validation sampeln dann random
        # validation size hyperparameter (10%)
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

    # TODO: hier mehrere dataloader returnen
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)