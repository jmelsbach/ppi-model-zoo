from lightning.pytorch.cli import LightningCLI
from ppi_zoo.datasets.RapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging

def cli_main() -> None:
    cli = LightningCLI(
        model_class=LSTMAWD,
        datamodule_class=RapppidDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={'callbacks': [StochasticWeightAveraging(0.05)]} # todo check if 0.05 is correct
    )

if __name__ == '__main__':
    cli_main()

