import torch
import pytorch_lightning as pl
from pathlib import Path

from dataset import AdobeFontDataset
from torchvision import transforms
from torchsummary import summary
import pfrl
from pfrl.experiments import LinearInterpolationHook
import hydra
from logging import getLogger
import shutil

from src.env import PatchSetsClassificationEnv
import src.env_model
import src.agent_model
from src.env_model_trainer import EnvModelTrainer


logger = getLogger(__name__)


def entropy_coef_setter(env, agent, value):
    agent.entropy_coef = value


# def evaluate(env, agent, evaluator, step, eval_score):
#     obs = env.reset()
#     action, _ = agent(obs)

#     accuracy
#     evaluator.tb_writer.add_scalar('eval/accuracy', accuracy, step)


@hydra.main(config_path='../config', config_name='config.yaml')
def main(config) -> None:
    all_done = False
    try:
        logger.info('\n' + hydra.utils.OmegaConf.to_yaml(config))

        shutil.copytree(
            Path(hydra.utils.get_original_cwd()) / 'src', Path.cwd() / 'copied' / 'src'
        )

        pl.seed_everything(0)

        if 'checkpoint_path' in config.env_model:
            env_model = src.env_model.EnvModel.load_from_checkpoint(
                config.env_model.checkpoint_path
            )
        else:
            env_model = src.env_model.EnvModel(**config.env_model)
        env_model = env_model.to(f'cuda:{config.gpu}')
        summary(env_model)
        env_model.eval()

        image_size = 100
        patch_size = 25
        obs_size = 1 + 1

        dataset = AdobeFontDataset(
            path='/dataset/AdobeFontCharImages',
            transform=transforms.ToTensor(),
            target_transform=lambda x: x['alphabet'],
            upper=True,
            lower=False,
        )

        env = PatchSetsClassificationEnv(
            dataset, env_model, patch_size, config.hparams.done_prob, collect_data=True
        )

        env_model_trainer = EnvModelTrainer(env_model)

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

        evaluation_hooks = [env_model_trainer]

        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=None,
            outdir='.',
            **config.experiment,
            step_hooks=step_hooks,
            evaluation_hooks=evaluation_hooks,
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
