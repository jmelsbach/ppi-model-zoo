from lightning.pytorch.cli import LightningCLI
from ppi_zoo.data.rapppid.OriginalRapppidDataset import OriginalRapppidDataModule
from ppi_zoo.models.rapppid.model import LSTMAWD
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as L

def cli_main() -> None:
    L.seed_everything(5353456, workers=True)
    cli = LightningCLI(
        model_class=LSTMAWD,
        datamodule_class=OriginalRapppidDataModule,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == '__main__':
    cli_main()

