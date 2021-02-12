from logging import getLogger
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.env import PatchSetBuffer
from src.env_model import EnvModel


logger = getLogger(__name__)


class EnvModelTrainer:
    def __init__(
        self,
        env_model: EnvModel,
        patch_set_buffer: PatchSetBuffer,
        batch_size: int,
        epochs: int,
        optimizer: torch.optim.Optimizer,
    ):
        self.env_model = env_model
        self.patch_set_buffer = patch_set_buffer

        self.epochs = epochs

        self.batch_size = batch_size
        self.optimizer = optimizer

        self.total_epoch = 0
        self._train_count = 0
        self._save_count = 0

    @staticmethod
    def collate_fn(batch: Iterable) -> tuple:
        x, y = list(zip(*batch))
        x = [torch.stack(i) for i in x]
        if all([x[0].shape == i.shape for i in x[1:]]):
            x = torch.stack(x)
        y = torch.tensor(y)
        return x, y

    def train(
        self,
        evaluator,
        step: int,
        dataloader,
        epochs: int,
    ):
        model = self.env_model
        model.train()

        total_loss = 0
        total_accuracy = 0
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            epoch_accuracy = 0
            for batch_index, data in enumerate(tqdm(dataloader, leave=False)):
                loss, accuracy = model.training_step(data, batch_index)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
            epoch_loss /= len(dataloader)
            epoch_accuracy /= len(dataloader)
            logger.info(
                f'epoch={epoch}, epoch_loss={epoch_loss}, epoch_accuracy={epoch_accuracy}'
            )
            evaluator.tb_writer.add_scalar(
                'env/epoch_loss', epoch_loss, self.total_epoch + epoch
            )
            evaluator.tb_writer.add_scalar(
                'env/epoch_accuracy', epoch_accuracy, self.total_epoch + epoch
            )
            total_loss += epoch_loss
            total_accuracy += epoch_accuracy
        self.total_epoch += self.epochs

        total_loss /= epoch + 1
        total_accuracy /= epoch + 1
        logger.info(f'step={step}, total_loss={total_loss}, total_loss={total_loss}')
        evaluator.tb_writer.add_scalar('env/train_loss', total_loss, step)
        evaluator.tb_writer.add_scalar('env/train_accuracy', total_accuracy, step)

        evaluator.tb_writer.add_scalar('env/total_epoch', self.total_epoch, step)
        evaluator.tb_writer.add_scalar('env/data_num', len(dataloader.dataset), step)
        evaluator.tb_writer.add_scalar('env/batch_num', len(dataloader), step)

        model.eval()

        save_dir = Path('env_model')
        save_dir.mkdir(exist_ok=True)
        if step > 100_000 * self._save_count:
            torch.save(model.state_dict(), str(save_dir / f'env_model_{step}.pth'))
            self._save_count += 1
        torch.save(model.state_dict(), str(save_dir / 'env_model_finish.pt'))

        self._train_count += 1

    def __call__(self, env, agent, evaluator, step, eval_score):
        train_dataset = self.patch_set_buffer()
        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        self.train(evaluator, step, train_dataloader, self.epochs)
