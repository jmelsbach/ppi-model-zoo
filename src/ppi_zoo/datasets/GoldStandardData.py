from torch.utils.data import Dataset
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class GoldStandardDataset(Dataset):

    def __init__(self, data_dir: str, tokenizer: object, max_len: int):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir).head(1000)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def _tokenize(self, sequence: str) -> dict:
        sequence = " ".join(sequence) # [TODO] sequence = re.sub(r"[UZOB]", "X", sequence_Example), sequence preprocessing based on tokenizer informations to still keep Goldstandarddatset generic
        tokens = self.tokenizer(
            sequence,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        tokens['input_ids'] = tokens['input_ids'].squeeze()
        tokens['attention_mask'] = tokens['input_ids'].squeeze()
        tokens['token_type_ids'] = tokens['input_ids'].squeeze()

        return tokens

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        target = torch.as_tensor(item['isInteraction'], dtype=torch.long)

        tokens_A = self._tokenize(sequence_A)
        tokens_B = self._tokenize(sequence_B)

        return tokens_A, tokens_B, target


class GoldStandardDataModule(pl.LightningDataModule):
    # TODO: hier url übergeben
    # url aus map rausnehmen für user
    # TODO: methoden beschreibung, parameter erklären
    def __init__(self, data_dir: str = '.data', batch_size: int = 2, num_workers: int = 4, tokenizer: object = None, max_len: int = 8, train_val_split: tuple = (1.0, 0.0)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_val_split = train_val_split

    # TODO: download data from cloud?
    def setup(self, stage=None):
        dataset = GoldStandardDataset(
            data_dir=self.data_dir, tokenizer=self.tokenizer, max_len=self.max_len)

        (train_size, val_size) = self.train_val_split
        all_train_indices = dataset.data.index[dataset.data['trainTest'] == 'train'].tolist(
        )

        if train_size == 1.0 and val_size == 0.0:
            train_indices, val_indices = all_train_indices, []
        else:
            train_indices, val_indices = train_test_split(
                all_train_indices, train_size=train_size, test_size=val_size, shuffle=True)

        test1_indices = dataset.data.index[dataset.data['trainTest'] == 'test1'].tolist(
        )
        test2_indices = dataset.data.index[dataset.data['trainTest'] == 'test2'].tolist(
        )

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test1_dataset = torch.utils.data.Subset(dataset, test1_indices)
        self.test2_dataset = torch.utils.data.Subset(dataset, test2_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return [
            DataLoader(self.test1_dataset, batch_size=self.batch_size,
                       shuffle=False, num_workers=self.num_workers),
            DataLoader(self.test2_dataset, batch_size=self.batch_size,
                       shuffle=False, num_workers=self.num_workers),
        ]
