import shutil
from logging import getLogger
from pathlib import Path

import hydra
import pfrl
import pytorch_lightning as pl
import torch
from pfrl.experiments import LinearInterpolationHook
from torchsummary import summary
from torchvision import transforms

from dataset import AdobeFontDataset
from src.agent_model import AgentModel
from src.env import PatchSetBuffer, PatchSetsClassificationEnv
from src.env_model import EnvModel
from src.env_model_evaluator import EnvModelEvaluator
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

        if config.gpu is not None:
            device = torch.device(f'cuda:{config.gpu}')
        else:
            device = torch.device(f'cpu')

        if 'checkpoint_path' in config.env_model and False:
            env_model = EnvModel.load_from_checkpoint(config.env_model.checkpoint_path)
        else:
            env_model = EnvModel(**config.env_model)
        env_model: EnvModel = env_model.to(device)
        summary(env_model)
        env_model.eval()

        train_patch_set_buffer = PatchSetBuffer(**config.patch_set_buffer.train)
        valid_patch_set_buffer = PatchSetBuffer(**config.patch_set_buffer.valid)

        train_dataset = AdobeFontDataset(
            transform=transforms.ToTensor(), **config.dataset.train
        )
        valid_dataset = AdobeFontDataset(
            transform=transforms.ToTensor(), **config.dataset.valid
        )

        train_env = PatchSetsClassificationEnv(
            dataset=train_dataset,
            env_model=env_model,
            **config.env,
            patch_set_buffer=train_patch_set_buffer,
        )
        valid_env = PatchSetsClassificationEnv(
            dataset=valid_dataset,
            env_model=env_model,
            **config.env,
            patch_set_buffer=valid_patch_set_buffer,
        )

        agent_model = AgentModel(**config.agent_model)
        summary(agent_model)

        env_optimizer = torch.optim.Adam(env_model.parameters(), **config.optim.env)
        agent_optimizer = torch.optim.Adam(
            agent_model.parameters(), **config.optim.agent
        )

        env_model_trainer = EnvModelTrainer(
            env_model=env_model,
            patch_set_buffer=train_patch_set_buffer,
            optimizer=env_optimizer,
            **config.env_model_trainer,
        )
        env_model_evaluator = EnvModelEvaluator(
            env_model=env_model,
            patch_set_buffer=valid_patch_set_buffer,
            **config.env_model_evaluator,
        )

        agent = pfrl.agents.PPO(
            model=agent_model, optimizer=agent_optimizer, **config.agent
        )

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

        evaluation_hooks = [env_model_trainer, env_model_evaluator]

        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            eval_env=valid_env,
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
