from lightning.pytorch.cli import LightningCLI
from ppi_zoo.datasets.RapppidDataset import RapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch.loggers import WandbLogger # TODO: do we need to import this?
import lightning.pytorch as L

def cli_main() -> None:
    L.seed_everything(5353456, workers=True)
    cli = LightningCLI(
        model_class=LSTMAWD,
        datamodule_class=RapppidDataModule,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == '__main__':
    cli_main()