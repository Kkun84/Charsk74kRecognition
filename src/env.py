import random
import gym
import random
import torch
from torch import Tensor
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
        feature_n: int,
        done_loss: float,
    ):
        self.action_space = None
        self.observation_space = None

        for attr in ['encode', 'pool', 'decode']:
            assert hasattr(model, attr), attr

        self.dataset = dataset
        self.model = model
        self.patch_size = patch_size
        self.feature_n = feature_n
        self.done_loss = done_loss

    @staticmethod
    def crop(image: Tensor, x: int, y: int, patch_size: int) -> Tensor:
        assert image.dim() == 3
        patch = image[:, y : y + patch_size, x : x + patch_size]
        assert patch.shape == torch.Size(
            [1, *[patch_size] * 2]
        ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
        # assert image.dim() == 3
        # image = nn.ConstantPad2d(patch_size // 2, 0)(image)
        # patch = image[:, y : y + patch_size, x : x + patch_size]
        # assert patch.shape == torch.Size(
        #     [1, *[patch_size] * 2]
        # ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
        return patch

    @staticmethod
    def make_observation(image: Tensor, feature: Tensor) -> Tensor:
        assert image.dim() == 3
        assert feature.dim() == 1
        observation = torch.cat(
            [
                image,
                *feature.squeeze(0)[:, None, None, None].expand(
                    len(feature), *image.shape
                ),
            ]
        )
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
                    'loss',
                    'reward',
                ]
            }

            if data_index is None:
                data = random.choice(self.dataset)
            else:
                data = self.dataset[data_index]
            self.data = (data[0].to(self.model.device), data[1])
            observation = self.make_observation(
                self.data[0].to(self.model.device),
                torch.zeros([self.feature_n], device=self.model.device),
            )
            self.last_loss = None
            self.step_count = 0
            # observation = torch.rand([65, 100, 100]).numpy()
            self.trajectory['observation'].append(observation)
            return observation

    def step(self, action: int) -> tuple:
        with torch.no_grad():
            image, target = self.data

            self.trajectory['action'].append(action)
            action_x = action % (image.shape[2] - self.patch_size + 1)
            action_y = action // (image.shape[2] - self.patch_size + 1)

            patch = self.crop(image, action_x, action_y, self.patch_size)
            self.trajectory['patch'].append(patch.detach().clone())

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

            y_hat = self.model.decode(feature)

            observation = self.make_observation(image, feature[0])
            self.trajectory['observation'].append(observation)

            loss = F.cross_entropy(
                y_hat,
                torch.tensor([target], dtype=torch.long, device=self.model.device),
            ).item()
            self.trajectory['loss'].append(loss)

            done = loss < self.done_loss

            if self.last_loss is None:
                self.last_loss = F.cross_entropy(
                    torch.zeros_like(y_hat),
                    torch.tensor([target], dtype=torch.long, device=self.model.device),
                ).item()
            if not done:
                reward = self.last_loss - loss
            else:
                reward = self.last_loss
            self.trajectory['reward'].append(reward)
            self.last_loss = loss
            self.step_count += 1
            # observation = torch.rand([65, 100, 100]).numpy()
            # reward = torch.randn([1]).item()
            # done = torch.randint(0, 2, [1]).item()
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

    # pl.seed_everything(0)

    model = src.env_model.Model.load_from_checkpoint(
        checkpoint_path='/workspace/epoch=1455.ckpt'
    )
    hparams = model.hparams
    summary(model)
    model.eval()

    image_size = 100
    patch_size = hparams.patch_size
    feature_n = hparams.feature_n
    obs_size = feature_n + 1
    n_actions = (image_size - patch_size) ** 2
    done_loss = 0.3

    dataset = [
        (
            torch.rand(1, image_size, image_size) * 100,
            torch.randint(26, [1]),
        )
        for i in range(100)
    ]

    env = PatchSetsClassificationEnv(dataset, model, patch_size, feature_n, done_loss)

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
