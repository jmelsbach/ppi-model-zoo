from lightning.pytorch.cli import LightningCLI
from ...src.ppi_zoo.datasets.StepGoogleDriveData import StepGoogleDriveDataModule
from ...src.ppi_zoo.models.step.model import STEP
from lightning.pytorch.loggers import WandbLogger

def cli_main() -> None:
    cli = LightningCLI(
        model_class=STEP,
        datamodule_class=StepGoogleDriveDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    cli_main()