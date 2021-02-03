from logging import getLogger

from typing import Any, Dict, List, Iterable, Union
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from src.env import PatchSetsClassificationEnv
from src.env_model import EnvModel
from src.agent_model import AgentModel


logger = getLogger(__name__)


class EnvModelEvaluator:
    def __init__(
        self,
        env_model: EnvModel,
        batch_size: int,
    ):
        self.env_model = env_model
        self.batch_size = batch_size
        self.total_epoch = 0

    @staticmethod
    def collate_fn(batch: Iterable) -> tuple:
        x, y = list(zip(*batch))
        x = [torch.stack(i) for i in x]
        if all([x[0].shape == i.shape for i in x[1:]]):
            x = torch.stack(x)
        y = torch.tensor(y)
        return x, y

    def eval(
        self,
        env: PatchSetsClassificationEnv,
        agent: AgentModel,
        evaluator,
        step: int,
        dataloader,
    ):

        model = self.env_model
        model.eval()

        loss = 0
        with torch.no_grad():
            step_outputs = []
            for batch_index, data in enumerate(tqdm(dataloader)):
                step_outputs.append(model.validation_step(data, batch_index))
        statistics = model._epoch_end(step_outputs)
        logger.info(f'step={step}, valid_loss={statistics["loss"]}, accuracy={statistics["accuracy"]}')
        evaluator.tb_writer.add_scalar(
            'env/valid_loss', statistics['loss'], step
        )
        evaluator.tb_writer.add_scalar(
            'env/accuracy', statistics['accuracy'], step
        )

        evaluator.tb_writer.add_scalar('env/total_epoch', self.total_epoch, step)
        evaluator.tb_writer.add_scalar('env/data_num', len(dataloader.dataset), step)
        evaluator.tb_writer.add_scalar('env/batch_num', len(dataloader), step)

        if step % 100_000 == 0:
            torch.save(model.state_dict(), f'env_model_{step}.pth')
        torch.save(model.state_dict(), 'env_model_finish.pth')

    def __call__(self, env, agent, evaluator, step, eval_score):

        for i in env.dataset:

        eval_dataset = self.patch_set_buffer()
        eval_dataloader = DataLoader(
            eval_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

        self.eval(env, agent, evaluator, step, eval_dataloader)
