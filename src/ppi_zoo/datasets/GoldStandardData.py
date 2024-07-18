from torch.utils.data import Dataset
import pandas as pd
import lightning.pytorch as L
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import re
from ppi_zoo.utils.gold_standard_utils import create_test_split
import pickle as pkl


class GoldStandardDataset(Dataset):

    def __init__(self, data_dir: str, file_name: str, tokenizer: object, max_len: int, limit: int):
        self.data_dir = data_dir
        self.data = pd.read_csv(f'{data_dir}/{file_name}')
        self.limit = limit or len(self.data)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _tokenize(self, sequence: str) -> dict:
        sequence = " ".join(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        tokens = self.tokenizer(
            sequence,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        tokens['input_ids'] = tokens['input_ids'].squeeze()
        tokens['attention_mask'] = tokens['attention_mask'].squeeze()
        tokens['token_type_ids'] = tokens['token_type_ids'].squeeze()

        return tokens

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        target = torch.as_tensor(item['isInteraction'], dtype=torch.long)

        tokens_A = self._tokenize(sequence_A)
        tokens_B = self._tokenize(sequence_B)

        return tokens_A, tokens_B, target


class GoldStandardDataModule(L.LightningDataModule):
    # TODO: hier url übergeben
    # url aus map rausnehmen für user
    def __init__(
        self,
        data_dir: str = '.data',
        file_name: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        tokenizer: str = None,
        max_len: int = 1536,
        with_validation: bool = True,
        limit: int = None
    ):
        """
        Data Module for Gold Standard PPI Dataset
        Reference: https://github.com/Llannelongue/B4PPI/tree/main
        Args:
            data_dir: path to directory which contains data
            batch_size: number of observations in each batch
            num_workers: number of workers used for data loading
            tokenizer: tokenizer used to id each token and create attention mask
            max_len: maximum number of tokens
            with_validation: controls whether to create a validation set or not
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, do_lower_case=False
        )
        self.max_len = max_len
        self.with_validation = with_validation
        self.limit = limit

    # TODO: download data from cloud?
    def setup(self, stage=None):
        dataset = GoldStandardDataset(
            data_dir=self.data_dir, file_name=self.file_name, tokenizer=self.tokenizer, max_len=self.max_len, limit=self.limit
        )

        df = dataset.data
        train_df = df.loc[
            df['trainTest'] == 'train'
        ]

        if self.with_validation:
            with open(f'{self.data_dir}/listHubs_human_20p_v2-1.pkl', 'rb') as f:
                protein_hubs = pkl.load(f)
            train_df, val_df = create_test_split(
                train_df, protein_hubs=protein_hubs
            )
        else:
            train_df, val_df = train_df, pd.DataFrame()

        train_indices = train_df.index
        val_indices = val_df.index

        test1_indices = df.index[
            dataset.data['trainTest'] == 'test1'
        ].tolist()

        test2_indices = df.index[
            dataset.data['trainTest'] == 'test2'
        ].tolist()

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices[0:self.limit])
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices[0:self.limit])
        self.test1_dataset = torch.utils.data.Subset(dataset, test1_indices[0:self.limit])
        self.test2_dataset = torch.utils.data.Subset(dataset, test2_indices[0:self.limit])

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
