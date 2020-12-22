import torch
import pytorch_lightning as pl
from pathlib import Path

# import logging
import sys
from datetime import datetime
from dataset import AdobeFontDataset
from torchvision import transforms
from torchsummary import summary
import pfrl
import hydra
from logging import getLogger
from omegaconf import OmegaConf

from src.env import PatchSetsClassificationEnv
import src.env_model
import src.agent_model
import shutil


logger = getLogger(__name__)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    all_done = False
    try:
        logger.info('\n' + OmegaConf.to_yaml(config))

        shutil.copytree(
            Path(hydra.utils.get_original_cwd()) / 'src', Path.cwd() / 'copied' / 'src'
        )

        pl.seed_everything(0)

        model = src.env_model.Model.load_from_checkpoint(
            checkpoint_path=config.env.checkpoint_path
        )
        hparams = model.hparams
        summary(model)
        model.eval()

        image_size = 100
        patch_size = hparams.patch_size
        feature_n = hparams.feature_n

        obs_size = feature_n + 1
        n_actions = (image_size - patch_size) ** 2
        done_loss = 0.1

        # dataset = [
        #     (
        #         torch.rand(1, image_size, image_size) * 100,
        #         torch.randint(26, [1]),
        #     )
        #     for i in range(100)
        # ]
        dataset = AdobeFontDataset(
            path='/dataset/AdobeFontCharImages',
            transform=transforms.ToTensor(),
            target_transform=lambda x: x['alphabet'],
            upper=True,
            lower=False,
        )

        env = PatchSetsClassificationEnv(
            dataset, model, patch_size, feature_n, done_loss
        )

        agent_model = src.agent_model.Model(obs_size, n_actions, 64, patch_size)
        summary(agent_model)

        optimizer = torch.optim.Adam(agent_model.parameters(), **config.optim)

        agent = pfrl.agents.PPO(model=agent_model, optimizer=optimizer, **config.agent)

        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=None,
            outdir='.',
            **config.experiment,
        )
        logger.info('All done.')
        all_done = True
    finally:
        if all_done == False:
            path = Path.cwd()
            if 'outputs' in path.parts or 'multirun' in path.parts:
                logger.info(f'Rename directory name. "{path}" -> "{path}__interrupted"')
                path.rename(path.parent / (path.name + '__interrupted__'))


if __name__ == "__main__":
    main()
