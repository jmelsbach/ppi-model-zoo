from lightning.pytorch.cli import LightningCLI
from ppi_zoo.datasets.RapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch.loggers import WandbLogger

def cli_main() -> None:
    cli = LightningCLI(
        model_class=LSTMAWD,
        datamodule_class=RapppidDataModule,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == '__main__':
    cli_main()

