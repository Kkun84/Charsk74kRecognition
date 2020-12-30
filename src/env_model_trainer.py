from logging import getLogger

from typing import Any, Dict, List, Iterable, Union
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel
from tqdm import tqdm

logger = getLogger(__name__)


class EnvModelTrainer:
    def __init__(self, env_model: EnvModel, update_step: int):
        self.update_step = update_step

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

    def train(self, env: PatchSetsClassificationEnv, agent, step: int):
        if step % self.update_step != 0:
            return

        model = env.model
        dataset = env.patch_set_pool

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
        for batch_idx, data in tqdm(enumerate(dataloader)):
            loss = model.training_step(data, batch_idx)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(data)
        total_loss /= len(dataset)

        logger.info(total_loss)

        model.eval()

    def __call__(self, env, agent, step):
        return self.train(env, agent, step)
