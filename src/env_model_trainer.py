from logging import getLogger

from typing import Any, Dict, List, Iterable, Union
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader

from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel
from tqdm import tqdm


logger = getLogger(__name__)


class EnvModelTrainer:
    def __init__(self, env_model: EnvModel):
        self.batch_size = env_model.hparams.batch_size
        self.optimizer = env_model.configure_optimizers()

    @staticmethod
    def collate_fn(batch: Iterable) -> tuple:
        x, y = list(zip(*batch))
        x = [torch.stack(i) for i in x]
        if all([x[0].shape == i.shape for i in x[1:]]):
            x = torch.stack(x)
        y = torch.tensor(y)
        return x, y

    def train(
        self, env: PatchSetsClassificationEnv, agent, evaluator, step: int, eval_score
    ):

        model: EnvModel = env.model
        dataset: Dataset = list(env.make_dataset()) * 10

        dataloader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

        model.train()

        total_loss = 0
        print()
        for batch_idx, data in enumerate(tqdm(dataloader)):
            loss = model.training_step(data, batch_idx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(data[0])
        total_loss /= len(dataset)

        evaluator.tb_writer.add_scalar('env/train_loss', total_loss, step)

        logger.info(total_loss)

        model.eval()

        if step % 100_000 == 0:
            torch.save(model.state_dict(), f'env_model_{step}.pth')
        torch.save(model.state_dict(), 'env_model_finish.pth')

    def __call__(self, env, agent, evaluator, step, eval_score):
        self.train(env, agent, evaluator, step, eval_score)
