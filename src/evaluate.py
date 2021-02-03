from logging import getLogger

from typing import Any, Dict, List, Iterable, Union
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.env import PatchSetsClassificationEnv, PatchSetBuffer
from src.env_model import EnvModel


logger = getLogger(__name__)


class EvaluateAccuracy:
    def __init__(self):
        pass

    @staticmethod
    def collate_fn(batch: Iterable) -> tuple:
        x, y = list(zip(*batch))
        x = [torch.stack(i) for i in x]
        if all([x[0].shape == i.shape for i in x[1:]]):
            x = torch.stack(x)
        y = torch.tensor(y)
        return x, y

    def __call__(self, env, agent, evaluator, step, eval_score):
        pass
