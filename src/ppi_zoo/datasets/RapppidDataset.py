from torch.utils.data import Dataset
import pandas as pd
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader
from ppi_zoo.utils.gold_standard_utils import create_test_split
import pickle as pkl
import sentencepiece as sp
import numpy as np

class RapppidDataset(Dataset):
    def __init__(self, data_dir: str, file_name: str, tokenizer: object, limit: int, truncate_len: int):
        self.data = pd.read_csv(f'{data_dir}/{file_name}')
        self.limit = min(limit, len(self.data))

    def __len__(self):
        return len(self.data)

    def _tokenize(self, sequence: str, use_sentence_processor: bool = True, use_padding: bool = True) -> dict:
        tokens = sequence[:self.hparams.truncate_len]

        if use_sentence_processor:
            tokens = np.array(self.hparams.tokenizer.encode(tokens, enable_sampling=True, alpha=0.1, nbest_size=-1))
        if use_padding:
            pad_len = self.hparams.truncate_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), 'constant')

        return tokens

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        target = torch.as_tensor(item['isInteraction'], dtype=torch.long)

        tokens_A = self._tokenize(sequence_A)
        tokens_B = self._tokenize(sequence_B)

        return tokens_A, tokens_B, target
    

    
class RapppidDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = '.data',
        file_name: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        tokenizer_file: str = None,
        with_validation: bool = True,
        limit: int = None,
        truncate_len: int = None
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
        self.tokenizer = sp.SentencePieceProcessor(model_file=tokenizer_file)

    def setup(self, stage=None):
        dataset = RapppidDataset(
            data_dir=self.hparams.data_dir, file_name=self.hparams.file_name, tokenizer=self.tokenizer, limit=self.hparams.limit, truncate_len=self.hparams.truncate_len
        )

        df = dataset.data
        train_df = df.loc[
            df['trainTest'] == 'train'
        ]

        if self.hparams.with_validation:
            with open(f'{self.hparams.data_dir}/listHubs_human_20p_v2-1.pkl', 'rb') as f:
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

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices[0:self.hparams.limit])
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices[0:self.hparams.limit])
        self.test1_dataset = torch.utils.data.Subset(dataset, test1_indices[0:self.hparams.limit])
        self.test2_dataset = torch.utils.data.Subset(dataset, test2_indices[0:self.hparams.limit])
        print('setup ready')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return [
            DataLoader(self.test1_dataset, batch_size=self.hparams.batch_size,
                       shuffle=False, num_workers=self.hparams.num_workers),
            DataLoader(self.test2_dataset, batch_size=self.hparams.batch_size,
                       shuffle=False, num_workers=self.hparams.num_workers),
        ]