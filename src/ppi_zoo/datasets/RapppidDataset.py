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
    def __init__(self, data_dir: str, file_name: str, tokenizer: object, truncate_len: int):
        self.data_dir = data_dir
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.data = pd.read_csv(f'{data_dir}/{file_name}')
        self.truncate_len = truncate_len

    def __len__(self):
        return len(self.data)

    def _tokenize(self, sequence: str, use_sentence_processor: bool = True, use_padding: bool = True) -> dict:
        tokens = sequence[:self.truncate_len]

        if use_sentence_processor:
            tokens = np.array(self.tokenizer.encode(tokens, enable_sampling=True, alpha=0.1, nbest_size=-1))
        if use_padding:
            pad_len = self.truncate_len - len(tokens)
            tokens = np.pad(tokens, (0, pad_len), 'constant')

        return tokens

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        target = torch.as_tensor(item['isInteraction'], dtype=torch.long)

        tokens_A = torch.as_tensor(self._tokenize(sequence_A), dtype=torch.long) # todo: do we need long here?
        tokens_B = torch.as_tensor(self._tokenize(sequence_B), dtype=torch.long) # todo: do we need long here?

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
        truncate_len: int = None
    ):
        super().__init__()
        self.save_hyperparameters() # TODO: dont use save_hyperparameters
        self.tokenizer = sp.SentencePieceProcessor(model_file=tokenizer_file)

    def setup(self, stage=None):
        dataset = RapppidDataset(
            data_dir=self.hparams.data_dir, file_name=self.hparams.file_name, tokenizer=self.tokenizer, truncate_len=self.hparams.truncate_len
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

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test1_dataset = torch.utils.data.Subset(dataset, test1_indices)
        self.test2_dataset = torch.utils.data.Subset(dataset, test2_indices)

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