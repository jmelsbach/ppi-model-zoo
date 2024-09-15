<div align="center">    
 
# PPI Model Zoo 

</div>
 
## Description   
PPI Model Zoo is a framework for benchmarking sequence-based PPI prediction models. We integrate the dataset from Lannelongue and Inouye (2024) to enable comparing models on datasets which (1) have a small amount of protein overlap (T1), and (2) contain a small proportion of positive instances (T2). So far two models have been benchmarked using this framework: the STEP model by Madan et al. (2022) and the RAPPPID model by Szymborski and Emad (2022). The results of the benchmark can be found in the experiments folder under "combined_reports". The following sections will describe how to install, use and contribute to the framework.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [How to contribute](#how-to-contribute)

## Installation 
Until the framework is up on PyPi, this can either be installed via cloning the package:

```
git clone git@github.com:jmelsbach/ppi-model-zoo.git
```
or
```
git clone https://github.com/jmelsbach/ppi-model-zoo.git
```
then:

```
cd ppi-model-zoo
pip install -e .
```

or directly installed from github:
```
python -m pip install git+https://github.com/jmelsbach/ppi-model-zoo.git
```

## Usage

### Get the Data

All the necessary datasets can be downloaded here: https://drive.google.com/drive/folders/15dWIpmf1HAI2uCbe7AfCIUWQrVOCKPyq?usp=sharing

### Benchmark your own model

In order to benchmark your own model two requirements need to be fulfilled.

1. You need to define a class for the datamodule. This class must extend `GoldStandardDataModule` and implement the `transform_sequence` method. The `transform_sequence` takes a protein sequence string as input and should return a pytorch tensor. The tensor may include all the necessary information about the sequence that you need to make a prediction with your model. For example it may include the token ids of each tokenized amino acid and the respective attention masks (see `src/ppi_zoo/data/step/StepDataModule.py` as an example). It is also necessary to call the constructor of `GoldStandardDataModule` using `super().__init__(*args, **kwargs)`.

```python
from ppi_zoo.data.GoldStandardDataModule import GoldStandardDataModule
import torch

class YourDataModule(GoldStandardDataModule):

  def __init__(self, your_parameter, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    # set your parameters
    # anything you need to transform the sequence
    # e.g. tokenizers

    self.your_parameter = your_parameter

  def transform_sequence(self, sequence: str) -> torch.Tensor:
    # define how a protein sequence string should be transformed to
    # interface with your model
    return torch.Tensor()

```

2. Your LightningModule needs to extend `GoldStandardPPILightningModule` and implement the `_build_model` and `forward` methods. Your `_build_model` method should initialize all necessary components that you need for making a classification e.g. classification heads, tokenizers. The forward method accepts two protein sequences which are transformed as you defined within `transform_sequence` of the datamodule. It is also necessary to call the constructor of `GoldStandardPPILightningModule` at the beginning of your constructor using `super().__init__()`.

```python
from ppi_zoo.models.GoldStandardPPILightningModule import GoldStandardPPILightningModule
import torch

class YourLightningModule(GoldStandardPPILightningModule):

  def __init__(self, my_hyper_parameters) -> None:
    super().__init__()
    # set your hyperparameters
    self.my_hyper_parameter = my_hyper_parameter

    # call _build_model
    self._build_model()

  def _build_model(self) -> None:
    # define all the necessary components for your forwards method
    # e.g. classification heads, initialize tokenizers etc.
    pass

  def forward(self, sequence_A, sequence_B) -> torch.Tensor:
    # calculate your models prediction based on sequence_A and sequence_B
    return torch.Tensor()

  # configure your opimizers and define all other necessary lightning methods
  def configure_optimizers(self):
    pass
```

Afterwards you can train, test and make predictions with your model using [LightninCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html). Please refer to the documentation for more info. To see examples refer to the scripts directory. Necessary parameters for the CLI are `data.data_dir` and `data.file_name`. These should point to where you have downloaded the data and which dataset you want to use.

### Compare to other models

To compare the results of your model to already benchmarked models you can simply create a new folder inside the experiments diretory for your model (with a reports sub directory) and drag the output of your test run into the directory named after the used dataset (4hubs, human, yeast, yeast_4hubs). Afterwards just run the `create_combined_report.py` script. To see what this looks like for STEP and RAPPPID refer to the experiments directory.

### Access our pretrained models

The checkpoints of all pretrained models can be found here: https://drive.google.com/drive/folders/1brcWV8j7VR0a3Kh7TlsO8XJJfCg6VeW3?usp=sharing.
Simply download the checkpoint and pass the `ckpt_path` argument to your LightningCLI to test or predict with the model (see `scripts/step/test.sh` for an example).

## How to contribute

### Set up development environment

1. Clone the project
```bash
git clone https://github.com/jmelsbach/ppi-model-zoo
```

2. Install [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/installation/)

3. Install [poetry](https://python-poetry.org/docs/#installation)
4. Install dependencies
```bash
poetry install
```
5. Activate environment
```bash
poetry shell
```

### Contribute your own model

To contribute your own model you need to provide your `GoldStandardDataModule` and `GoldStandardPPILightningModule` as defined [here](#benchmark-your-own-model). Simply create a new directory with your models name under `src/ppi_zoo/models` and `src/ppi_zoo/data`. Paste your Lightni`GoldStandardPPILightningModule`ngModule and all other relevant classes into the model directory and your `GoldStandardDataModule` into the data directory. The default parameters of both your `GoldStandardPPILightningModule` and `GoldStandardDataModule` should be set to the optimal hyperparameters used for your final training run. Additionally, we would like you to contribute all training and testing scripts as well as your LightningCLI to the `scripts` directory. For this create a directory with your models' name. Lastly, you should contribute the results of the test run to the experiments directory (at least for the human dataset). To get an updated report run the `create_combined_report.py` script and contribute its output aswell.