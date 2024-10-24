from lightning.pytorch.cli import LightningCLI
from ppi_zoo.data.step.StepDataModule import StepDataModule
from ppi_zoo.models.step.model import STEP
from lightning.pytorch.loggers import WandbLogger

def cli_main() -> None:
    cli = LightningCLI(
        model_class=STEP,
        datamodule_class=StepDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    cli_main()

