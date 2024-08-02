from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import re
import os
import shutil
import zipfile
from rich.progress import Progress
import requests
from typing import Type, Union

# Tokenize data for STEP model
class StepGoogleDriveDataset(Dataset):
    def __init__(self, data, base_model, max_length=512, return_labels=True):
        self.data = data
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_length
        self.return_labels = return_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_a = self.data["sequenceA"].iloc[idx]
        sequence_b = self.data["sequenceB"].iloc[idx]
        label = self.data["class"].iloc[idx]

        if (
            "yarongef".lower() in self.base_model.lower()
            or "rostlab" in self.base_model.lower()
        ):
            # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            # See https://huggingface.co/Rostlab/prot_bert#preprocessing
            sequence_a = " ".join(sequence_a)
            sequence_b = " ".join(sequence_b)
            sequence_a = re.sub(r"[UZOB]", "X", sequence_a)
            sequence_b = re.sub(r"[UZOB]", "X", sequence_b)

        inputs_a = self.tokenizer(
            sequence_a,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs_b = self.tokenizer(
            sequence_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.return_labels:
            return (inputs_a, inputs_b, label)
        else:
            return (inputs_a, inputs_b)
        
class StepGoogleDriveDataModule(LightningDataModule):
    def __init__(
        self,
        test_path,
        train_path,
        val_path,
        batch_size: int = 256,
        num_workers: int = 8,
        base_model: str = "Rostlab/prot_bert_bfd",
        dataset: dict = {
            'url': 'https://drive.google.com/file/d/1AFSsJ3RWWoeHk09SDVTbScf9HemcTpFd/view?usp=sharing',
            'file_name': 'step_madan_train.zip',
            'name': 'step_madan_train'
        }
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_path = Path(test_path) 
        self.train_path = Path(train_path) 
        self.val_path = Path(val_path)
        self.base_model = base_model
        self.dataset_dir = Path.home() / ".exect-data"
        self.dataset = dataset

    def prepare_data(self):
        # Download and unzip dataset
        self.download_dataset(self.dataset, path=self.dataset_dir)
        self.unzip_dataset(self.dataset, path=self.dataset_dir)

    def download_dataset(self, dataset: dict, path: Type[Union[Path, str]] = Path.home() / ".exect-data", force_download=False):
        """
        It downloads a file from a URL and saves it to a specified path

        Args:
          dataset (dict): Dictionary containing the name and the url of a dataset.
          path (str): The path to the directory where the dataset will be downloaded. Defaults to ./datasets
          force_download: If True, the dataset will be downloaded even if it already exists in the path.
        Defaults to False
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        url = dataset['url']
        zip_file_name = dataset['file_name']
        dataset_name = dataset['name']
        
        if (Path(path) / dataset_name).exists() and not force_download:
            print(f"Dataset already exists.")
        elif "drive.google.com" in url:
            import gdown

            gdown.download(url, os.path.join(path, zip_file_name), quiet=False)
        else:
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = total_size_in_bytes // 1000

            with Progress(transient=True) as progress:
                download_task = progress.add_task(
                    f":arrow_down: Downloading {dataset_name}...", total=total_size_in_bytes
                )

                with open(os.path.join(path, zip_file_name), "wb") as file:
                    for data in response.iter_content(block_size):
                        progress.update(download_task, advance=block_size)
                        file.write(data)

    def unzip_dataset(self, dataset: dict, path: Type[Union[Path, str]] = Path.home() / ".exect-data"):
        zip_file_name = dataset['file_name']

        if isinstance(path, str):
            path = Path(path)

        if (path / zip_file_name).exists():
            with zipfile.ZipFile(os.path.join(path, zip_file_name), "r") as zip_ref:
                for member in zip_ref.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue

                    source = zip_ref.open(member)
                    os.makedirs(path / dataset['name'], exist_ok=True)
                    target = open(os.path.join(path / dataset['name'], filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

            os.remove(path / zip_file_name)

    def setup(self, stage: str = None):
        # Get the dataset
        data_path = self.dataset_dir / self.dataset['name']
        df = pd.read_csv(data_path / "train.csv", delimiter="\t")

        # Perform the train-val-test-split:
        train_size = int(len(df) * 0.5)
        val_size = int(len(df) * 0.2)
        test_size = len(df) - train_size - val_size

        train_df, val_df, test_df = random_split(df, [train_size, val_size, test_size])

        self.train_dataset = StepGoogleDriveDataset(
            train_df, self.base_model, max_length=512, return_labels=True
        )
        self.val_dataset = StepGoogleDriveDataset(
            val_df, self.base_model, max_length=512, return_labels=True
        )
        self.test_dataset = StepGoogleDriveDataset(
            test_df, self.base_model, max_length=512, return_labels=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
