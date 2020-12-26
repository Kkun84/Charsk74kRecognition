import torch
import pytorch_lightning as pl
from pathlib import Path

# import logging
from dataset import AdobeFontDataset
from torchvision import transforms
from torchsummary import summary
import pfrl
from pfrl.experiments import LinearInterpolationHook
import hydra
from logging import getLogger
from omegaconf import OmegaConf

from src.env import PatchSetsClassificationEnv
import src.env_model
import src.agent_model
import shutil


logger = getLogger(__name__)


def entropy_coef_setter(env, agent, value):
    agent.entropy_coef = value


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

        obs_size = 1 + 1
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

        env = PatchSetsClassificationEnv(dataset, model, patch_size, done_loss)

        # agent_model = src.agent_model.Model(obs_size, n_actions, patch_size)
        agent_model = src.agent_model.Model(obs_size, patch_size)
        summary(agent_model)

        optimizer = torch.optim.Adam(agent_model.parameters(), **config.optim)

        agent = pfrl.agents.PPO(model=agent_model, optimizer=optimizer, **config.agent)

        step_hooks = []
        if None not in [
            config.hparams.start_entropy_coef,
            config.hparams.end_entropy_coef,
        ]:
            kwargs = dict(
                total_steps=config.experiment.steps
                * config.hparams.total_steps_entropy_coef,
                start_value=config.hparams.start_entropy_coef,
                stop_value=config.hparams.end_entropy_coef,
                setter=entropy_coef_setter,
            )
            logger.info(
                f'LinearInterpolationHook({", ".join([f"{k}={v}" for k, v in kwargs.items()])})'
            )
            step_hooks.append(LinearInterpolationHook(**kwargs))

        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=None,
            outdir='.',
            **config.experiment,
            step_hooks=step_hooks,
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
