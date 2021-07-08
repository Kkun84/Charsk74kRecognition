import shutil
from logging import INFO as LOG_LEVEL, Logger
from logging import FileHandler, StreamHandler, getLogger
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torchinfo
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from plmodule import DataModule, LightningModule


logger = getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def get_command() -> str:
    import sys

    return ' '.join(sys.argv)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    all_done = False
    try:
        set_pytorch_lightning_logging(logger)

        logger.info('Loaded "config.yaml"\n' + OmegaConf.to_yaml(config))

        shutil.copytree(
            Path(hydra.utils.get_original_cwd()) / 'src',
            Path.cwd() / 'snapshot' / 'src',
        )

        with open('command.txt', 'w') as file:
            file.write(get_command())

        pl.seed_everything(config.seed)

        trainer = pl.Trainer(
            **config.trainer,
            callbacks=[
                ModelCheckpoint(**config.model_checkpoint),
                EarlyStopping(**config.early_stopping),
            ]
            + [hydra.utils.instantiate(i) for i in config.callbacks],
            logger=[hydra.utils.instantiate(i) for i in config.loggers],
        )

        lightning_module = LightningModule(**config.lightning_module)
        logger.info(
            'torchinfo.summary\n'
            + str(torchinfo.summary(lightning_module, [2, 3, 100, 100], verbose=0))
        )

        data_module = DataModule(**config.data_module)

        assert lightning_module.hparams.num_classes == data_module.num_classes

        trainer.fit(model=lightning_module, datamodule=data_module)

        results = trainer.test(
            model=lightning_module, datamodule=data_module, ckpt_path='best'
        )
        logger.info(f'Test results: {results}')

        logger.info('All done.')
        all_done = True
    finally:
        if all_done == False:
            cwd_path = Path.cwd()
            if 'outputs' in cwd_path.parts or 'multirun' in cwd_path.parts:
                suffix = '__interrupted__'
                logger.info(
                    f'Rename directory name. "{cwd_path}" -> "{cwd_path}{suffix}"'
                )
                cwd_path.rename(str(cwd_path) + suffix)


def set_pytorch_lightning_logging(hydra_logger: Logger):
    import sys

    stream_handlers = [
        i for i in hydra_logger.root.handlers if type(i) == StreamHandler
    ]
    assert len(stream_handlers) == 1
    file_handlers = [i for i in hydra_logger.root.handlers if type(i) == FileHandler]
    assert len(file_handlers) == 1

    pl_logger = getLogger('pytorch_lightning')
    pl_logger.setLevel(LOG_LEVEL)

    pl_logger.addHandler(file_handlers[0])
    pl_logger.addHandler(stream_handlers[0])


if __name__ == "__main__":
    main()
