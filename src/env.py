import random
import gym
import random
import torch
from torch import Tensor
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Iterable
import pytorch_lightning as pl


class PatchSetsClassificationEnv(gym.Env):
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        patch_size: int,
        done_prob: float = 1,
    ):
        self.action_space = gym.spaces.Discrete((100 - patch_size + 1) ** 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[2, 100, 100])

        self.dataset = dataset
        self.model = model
        self.patch_size = patch_size
        self.done_prob = done_prob

        self.patch_set_pool = []

    def make_dataset(self):
        self.patch_set_pool
        return

    @staticmethod
    def crop(image: Tensor, x: int, y: int, patch_size: int) -> Tensor:
        assert image.dim() == 3
        patch = image[:, y : y + patch_size, x : x + patch_size]
        assert patch.shape == torch.Size(
            [1, *[patch_size] * 2]
        ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
        return patch

    @staticmethod
    def make_mask(
        image: Tensor,
        x: int = None,
        y: int = None,
        patch_size: int = None,
        mask: Tensor = None,
    ) -> Tensor:
        assert image.dim() == 3
        if mask is None:
            mask = torch.zeros([1, *image.shape[1:]], device=image.device)
        else:
            mask = mask.clone()
        if x is not None and y is not None and patch_size is not None:
            assert mask.dim() == 3
            assert image.shape[1:] == mask.shape[1:]
            assert 0 <= y <= image.shape[1] - patch_size
            assert 0 <= x <= image.shape[2] - patch_size
            mask[:, y : y + patch_size, x : x + patch_size] += 1
        elif not (x is None and y is None and patch_size is None):
            assert False
        return mask

    @staticmethod
    def make_observation(image: Tensor, mask: Tensor) -> Tensor:
        assert image.dim() == 3
        assert mask.dim() == 3
        assert image.shape[1:] == mask.shape[1:]
        observation = torch.cat([image, mask])
        return observation

    def reset(self, data_index: int = None) -> Tensor:
        with torch.no_grad():
            self.trajectory = {
                i: []
                for i in [
                    'observation',
                    'action',
                    'patch',
                    'feature_set',
                    'likelihood',
                    'loss',
                    'reward',
                ]
            }

            if data_index is None:
                data = random.choice(self.dataset)
            else:
                data = self.dataset[data_index]

            self.data = (data[0].to(self.model.device), data[1])

            image = self.data[0]
            self.mask = self.make_mask(image)
            observation = self.make_observation(image, self.mask)

            self.last_likelihood_advantage = 0
            self.step_count = 0
            self.trajectory['observation'].append(observation)

            self.patch_set_pool.append(([], data[1]))

            return observation

    def step(self, action: int) -> tuple:
        with torch.no_grad():
            image, target = self.data

            self.trajectory['action'].append(action)
            action_x = action % (image.shape[2] - self.patch_size + 1)
            action_y = action // (image.shape[2] - self.patch_size + 1)

            patch = self.crop(image, action_x, action_y, self.patch_size)
            self.trajectory['patch'].append(patch.detach().clone())
            self.patch_set_pool[-1][0].append(patch.detach().cpu().clone())

            feature_set = self.model.encode(patch[None, None])
            if self.step_count > 0:
                feature_set = torch.cat(
                    [
                        feature_set,
                        self.trajectory['feature_set'][-1],
                    ],
                    1,
                )

            self.trajectory['feature_set'].append(feature_set.detach().clone())

            feature = self.model.pool(feature_set)

            likelihood = self.model.decode(feature)
            self.trajectory['likelihood'].append(likelihood.detach().clone())

            image = self.data[0]
            self.mask = self.make_mask(
                image, action_x, action_y, self.patch_size, self.mask
            )
            observation = self.make_observation(image, self.mask)
            self.trajectory['observation'].append(observation)

            loss = F.cross_entropy(
                likelihood,
                torch.tensor([target], dtype=torch.long, device=self.model.device),
            ).item()
            self.trajectory['loss'].append(loss)

            prob = likelihood.softmax(1)[0]
            done = prob[target] > self.done_prob

            if not done:
                likelihood_advantage = (
                    prob[target]
                    - torch.cat([prob[:target], prob[target + 1 :]], 0).max()
                ).item()
                reward = likelihood_advantage - self.last_likelihood_advantage
            else:
                likelihood_advantage = 1
                reward = likelihood_advantage - self.last_likelihood_advantage
            self.last_likelihood_advantage = likelihood_advantage
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
    import src.env_model
    import src.agent_model
    from dataset import AdobeFontDataset
    from torchvision import transforms
    from torchsummary import summary
    import pfrl

    pl.seed_everything(0)

    model = src.env_model.EnvModel(
        patch_size=25, hidden_n=1, feature_n=16, output_n=26, pool_mode='sum'
    )
    hparams = model.hparams
    summary(model)
    model.eval()

    image_size = 100
    patch_size = hparams.patch_size
    obs_size = 2
    n_actions = (image_size - patch_size) ** 2
    done_prob = 0.3

    dataset = [
        (
            torch.rand(1, image_size, image_size) * 100,
            torch.randint(26, [1]),
        )
        for i in range(100)
    ]

    env = PatchSetsClassificationEnv(dataset, model, patch_size)

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
    pass
