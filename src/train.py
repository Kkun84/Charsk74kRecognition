import shutil
from logging import getLogger, FileHandler
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torchinfo
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from plmodule import DataModule, LightningModule


logger = getLogger(__name__)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    all_done = False
    try:
        logger.info('\n' + OmegaConf.to_yaml(config))

        shutil.copytree(
            Path(hydra.utils.get_original_cwd()) / 'src',
            Path.cwd() / 'snapshot' / 'src',
        )

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

        trainer.fit(model=lightning_module, datamodule=data_module)

        trainer.test()

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


if __name__ == "__main__":
    main()
