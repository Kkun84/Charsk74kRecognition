import random
from logging import getLogger
from typing import Dict, Iterable, List, Optional, Tuple, Type

import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch._C import device
from torch.utils.data import Dataset

from src.env_model import EnvModel

logger = getLogger(__name__)


class PatchSetBuffer:
    def __init__(self, use_n_steps: int = 1):
        assert use_n_steps > 0
        self.use_n_steps = use_n_steps
        self.data: List[Tuple[List[Tensor], int]] = []
        self._data: List[List[Tuple[List[Tensor], int]]] = []

    def __call__(self) -> List[Tuple[List[Tensor], int]]:
        self._data.append(self.data[:-1])
        self.data = self.data[-1:]
        dataset = sum(self._data, [])
        if len(self._data) >= self.use_n_steps:
            del self._data[0]
        return dataset

    def __len__(self) -> int:
        length = len(sum(self._data, [])) + len(self.data[:-1])
        return length

    def reset(self, x: Type[None], y: int) -> None:
        assert x is None
        assert y is not None
        self.data.append(([], y))

    def step(self, x: Tensor, y: Type[None]) -> None:
        assert x is not None
        assert y is None
        self.data[-1][0].append(x)


class PatchSetsClassificationEnv(gym.Env):
    def __init__(
        self,
        dataset: Dataset,
        env_model: EnvModel,
        patch_size: Dict[str, int],
        done_threshold: float = 0,
        patch_set_buffer: Optional[PatchSetBuffer] = None,
    ):
        assert {'x', 'y'} == set(patch_size.keys())

        self.action_space = gym.spaces.Discrete((100 - 25 + 1) ** 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[2, 100, 100])

        self.dataset = dataset
        self.env_model = env_model
        self.patch_size = patch_size
        self.done_threshold = done_threshold

        self._random_index = []

        self.patch_set_buffer = patch_set_buffer

    @staticmethod
    def crop(
        image: Tensor, x: int, y: int, patch_size: Dict[str, int], copy: bool = True
    ) -> Tensor:
        assert image.dim() == 3
        patch = image[:, y : y + patch_size['y'], x : x + patch_size['x']]
        if copy:
            patch = patch.clone()
        assert patch.shape == torch.Size(
            [1, patch_size['y'], patch_size['x']]
        ), f"{patch.shape} == {torch.Size([1, patch_size['y'], patch_size['x']])}"
        return patch

    @staticmethod
    def make_patch(image: Tensor, x: int, y: int, patch_size: Dict[str, int]) -> Tensor:
        patch = PatchSetsClassificationEnv.crop(image, x, y, patch_size)
        return patch
        canvas = torch.zeros(
            [1, image.shape[1], image.shape[2]], dtype=torch.float, device=image.device
        )
        patch = PatchSetsClassificationEnv.crop(image, x, y, patch_size)
        PatchSetsClassificationEnv.crop(canvas, x, y, patch_size)[:] = patch
        return canvas

    @staticmethod
    def make_patched_area(
        height: int,
        width: int,
        x: int,
        y: int,
        patch_size: Dict[str, int],
        device=None,
    ) -> Tensor:
        mask = torch.zeros([1, height, width], dtype=torch.bool, device=device)
        assert 0 <= x <= width - patch_size['x']
        assert 0 <= y <= height - patch_size['y']
        mask[:, y : y + patch_size['y'], x : x + patch_size['x']] = True
        return mask

    @staticmethod
    def make_observation(image: Tensor, mask: Tensor) -> Tensor:
        assert image.dim() == 3
        assert mask.dim() == 3
        assert image.shape[1:] == mask.shape[1:]
        observation = torch.cat([image, mask])
        return observation

    def reset(self, data_index: int = None) -> Tensor:
        device = self.env_model.device
        with torch.no_grad():
            self.trajectory = {
                i: []
                for i in [
                    'observation',
                    'action',
                    'patch',
                    'feature_set',
                    'output',
                    'loss',
                    'reward',
                ]
            }

            if data_index is None:
                if not self._random_index:
                    self._random_index = random.sample(
                        range(len(self.dataset)), k=len(self.dataset)
                    )
                data = self.dataset[self._random_index.pop()]
            else:
                data = self.dataset[data_index]

            self.data = (data[0].to(device), data[1])

            image = self.data[0]
            self.patched_area = torch.zeros(
                [1, image.shape[1], image.shape[2]], dtype=torch.float, device=device
            )
            observation = self.make_observation(image, self.patched_area)

            self.last_output_advantage = 0
            self.step_count = 0
            self.trajectory['observation'].append(observation)

            if self.patch_set_buffer is not None:
                self.patch_set_buffer.reset(x=None, y=data[1])

            return observation

    def step(self, action: int) -> tuple:
        device = self.env_model.device
        with torch.no_grad():
            image, target = self.data

            self.trajectory['action'].append(action)
            action_x = action % (image.shape[2] - self.patch_size['x'] + 1)
            action_y = action // (image.shape[2] - self.patch_size['y'] + 1)

            patch = PatchSetsClassificationEnv.make_patch(
                image, action_x, action_y, self.patch_size
            )
            self.trajectory['patch'].append(patch.detach().clone())

            if self.patch_set_buffer is not None:
                self.patch_set_buffer.step(patch.detach().cpu().clone(), None)

            feature_set = self.env_model.encode(patch[None, None])
            if self.step_count > 0:
                feature_set = torch.cat(
                    [
                        feature_set,
                        self.trajectory['feature_set'][-1],
                    ],
                    1,
                )

            self.trajectory['feature_set'].append(feature_set.detach().clone())

            feature = self.env_model.pool(feature_set)

            output = self.env_model.decode(feature)
            self.trajectory['output'].append(output.detach().clone())

            image = self.data[0]
            self.patched_area = self.patched_area + self.make_patched_area(
                image.shape[1],
                image.shape[2],
                action_x,
                action_y,
                self.patch_size,
                device,
            )
            observation = self.make_observation(image, self.patched_area)
            self.trajectory['observation'].append(observation)

            loss = F.cross_entropy(
                output,
                torch.tensor([target], dtype=torch.long, device=device),
            ).item()
            self.trajectory['loss'].append(loss)

            probability = output.softmax(1)[0]
            # done = probability[target] > self.done_threshold
            done = probability.max() > self.done_threshold

            if not done:
                output_advantage = (
                    probability[target]
                    - torch.cat(
                        [probability[:target], probability[target + 1 :]], 0
                    ).max()
                ).item()
            elif probability[target] > self.done_threshold:
                output_advantage = 1
            elif probability.max() > self.done_threshold:
                output_advantage = -1
            else:
                assert False
            reward = output_advantage - self.last_output_advantage

            self.last_output_advantage = output_advantage
            self.trajectory['reward'].append(reward)
            self.step_count += 1
            return observation, reward, done, {}

    # def render(self, mode="human", close=False):
    #     pass

    # def close(self):
    #     pass

    # def seed(self, seed=None):
    #     pass


if __name__ == "__main__":
    from torchsummary import summary

    import src.agent_model
    import src.env_model

    pl.seed_everything(0)

    env_model = src.env_model.EnvModel(
        hidden_n=1, feature_n=16, output_n=26, pool_mode='sum'
    )
    summary(env_model)
    env_model.eval()

    image_size = 100
    patch_size = {'x': 9, 'y': 8}
    obs_size = 2
    done_threshold = 0.3

    dataset = [
        (
            torch.rand(1, image_size, image_size) * 100,
            torch.randint(26, [1]),
        )
        for i in range(100)
    ]

    patch_set_buffer = PatchSetBuffer()

    env = PatchSetsClassificationEnv(
        dataset, env_model, patch_size, 0.5, patch_set_buffer
    )

    for _ in range(4):
        state = env.reset()
        print(state.shape)
        action = 0
        state, reward, done, _ = env.step(action)
        print(state.shape)
        print(reward)
        print(done)
        action = 1
        state, reward, done, _ = env.step(action)
        print(state.shape)
        print(reward)
        print(done)
