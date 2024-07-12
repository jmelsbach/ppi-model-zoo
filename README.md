<div align="center">    
 
# PPI Model Zoo

## Model Execution
- "poetry shell" -> Activate poetry env
- "nvidia-smi" -> check GPU utilization
- "bash scripts/step/test.sh" -> start testing script
- "bash scripts/step/fit.sh" -> start testing script
- slurm:
  - "squeue": jobs listen
  - "scancel <id>": job canceln
  - "sbatch fit_slurm.sh": job starten (vorher poetry shell machen bzw ins script einbauen)
- "fit.sh --debug" -> for Debugging 

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)]()
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)]()
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)]()
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)]()  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/jmelsbach/ppi-model-zoo

# install poetry
LINK TO POETRY DOCU

# install dependencies 
poetry install

 ```   
 Next, navigate to script directory and run any script.   
 ```bash
# module folder
cd ppi_zoo/src/scripts
python script.py

```

## Imports (TODO: update later)
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   