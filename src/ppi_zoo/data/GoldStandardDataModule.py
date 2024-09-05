import pandas as pd
import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader
from ppi_zoo.utils.gold_standard_utils import create_test_split
from ppi_zoo.data.GoldStandardDataset import GoldStandardDataset
import pickle as pkl


class GoldStandardDataModule(L.LightningDataModule):
    # TODO: hier url übergeben
    # url aus map rausnehmen für user
    def __init__(
        self,
        data_dir: str = '.data',
        file_name: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        with_validation: bool = True,
        limit: int = None,
    ):
        """
        Data Module for Gold Standard PPI Dataset
        Reference: https://github.com/Llannelongue/B4PPI/tree/main
        Args:
            data_dir: path to directory which contains data
            batch_size: number of observations in each batch
            num_workers: number of workers used for data loading
            with_validation: controls whether to create a validation set or not
        """
        super().__init__()
        self.data_dir = data_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_validation = with_validation
        self.limit = limit

    def transform_sequence(self, sequence: str) -> torch.Tensor:
        raise NotImplementedError

    # TODO: download data from cloud?
    def setup(self, stage=None):
        dataset = GoldStandardDataset(
            data_dir=self.data_dir,
            file_name=self.file_name,
            transform_sequence=self.transform_sequence,
            limit=self.limit
        )

        df = dataset.data
        train_df = df.loc[
            df['trainTest'] == 'train'
        ]

        if self.with_validation:
            with open(f'{self.data_dir}/listHubs_human_20p_v2-1.pkl', 'rb') as f: #
                protein_hubs = pkl.load(f)
            train_df, val_df = create_test_split(
                train_df, protein_hubs=protein_hubs
            )
        else:
            train_df, val_df = train_df, pd.DataFrame()

        train_indices = train_df.index
        val_indices = val_df.index

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices[0:self.limit])
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices[0:self.limit])

        train_test_labels = dataset.data['trainTest'].unique()

        if 'test' in train_test_labels:
            test_indices = df.index[
                dataset.data['trainTest'] == 'test'
            ].tolist()
            self.test_dataset = torch.utils.data.Subset(dataset, test_indices[0:self.limit])

        if 'test1' in train_test_labels:
            test1_indices = df.index[
                dataset.data['trainTest'] == 'test1'
            ].tolist()
            self.test1_dataset = torch.utils.data.Subset(dataset, test1_indices[0:self.limit])

        if 'test2' in train_test_labels:
            test2_indices = df.index[
                dataset.data['trainTest'] == 'test2'
            ].tolist()
            self.test2_dataset = torch.utils.data.Subset(dataset, test2_indices[0:self.limit])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        
        return [
            DataLoader(
                self.test1_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            ),
            DataLoader(
                self.test2_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            ),
        ]
