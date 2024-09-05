from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Callable
import lightning.pytorch as L

class GoldStandardDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        file_name: str,
        transform_sequence: Callable[[L.LightningDataModule, str], torch.Tensor],
        limit: int
    ):
        self.data_dir = data_dir
        self.data = pd.read_csv(f'{data_dir}/{file_name}')
        self.limit = limit or len(self.data)
        self.transform_sequence = transform_sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        sequence_A = item['sequence_A']
        sequence_B = item['sequence_B']
        target = torch.as_tensor(item['isInteraction'], dtype=torch.long)

        tokens_A = self.transform_sequence(sequence_A)
        tokens_B = self.transform_sequence(sequence_B)

        return tokens_A, tokens_B, target
