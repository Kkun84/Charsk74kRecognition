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


class PatchSetsClassification(gym.Env):
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
        image = nn.ConstantPad2d(patch_size // 2, 0)(image)
        patch = image[:, x : x + patch_size, y : y + patch_size]
        assert patch.shape == torch.Size(
            [1, *[patch_size] * 2]
        ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
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
            self.trajectory = {i: [] for i in ['action', 'patch', 'feature']}

            if data_index is None:
                self.data = random.choice(self.dataset)
            else:
                self.data = self.dataset[data_index]
            observation = self.make_observation(
                self.data[0], torch.zeros([self.feature_n])
            )
            self.last_loss = F.binary_cross_entropy_with_logits(
                torch.zeros_like(self.data[1])[None], self.data[1][None]
            ).item()
        self.step_count = 0
        return observation

    def step(self, action: Iterable) -> tuple:
        with torch.no_grad():
            image = self.data[0]

            self.trajectory['action'].append(action)
            action_x = action // image.shape[1]
            action_y = action % image.shape[2]

            patch = self.crop(image, action_x, action_y, self.patch_size)
            self.trajectory['patch'].append(patch)

            feature_set = self.model.encode(patch[None, None])
            if self.step_count > 0:
                feature_set = torch.cat(
                    [feature_set, self.trajectory['feature'][-1].unsqueeze(1)], 1
                )

            feature = self.model.pool(feature_set)
            self.trajectory['feature'].append(feature)

            y_hat = self.model.decode(feature)

            observation = self.make_observation(self.data[0], feature[0])

            loss = F.binary_cross_entropy_with_logits(y_hat, self.data[1][None]).item()

            done = loss < self.done_loss

            if not done:
                reward = self.last_loss - loss
            else:
                reward = self.last_loss
            self.last_loss = loss
            if done:
                reward = reward
        self.step_count += 1
        return observation, reward, done, {}

    # def render(self, mode="human", close=False):
    #     pass

    # def close(self):
    #     pass

    # def seed(self, seed=None):
    #     pass


if __name__ == "__main__":
    import src.model

    image_size = 100
    patch_size = 3
    input_n, hidden_n, feature_n, output_n = patch_size ** 2, 4, 4, 4
    obs_size, n_actions = feature_n + 1, image_size ** 2
    done_loss = 0.14

    pl.seed_everything(0)

    model = src.model.Model(input_n, hidden_n, feature_n, output_n)

    dataset = [
        (
            torch.rand(1, image_size, image_size) * 100,
            torch.eye(output_n)[torch.randint(output_n, [1]) * 0][0],
        )
        for i in range(1000)
    ]

    env = PatchSetsClassification(dataset, model, patch_size, feature_n, done_loss)

    state = env.reset()

    print(state.shape)

    action = 0

    state, reward, done, _ = env.step(action)

    print(state.shape)
    print(reward)
    print(done)

    print("@" * 80)

    import pfrl

    ppo = src.model.PPO(obs_size, n_actions, hidden_n)

    optimizer = torch.optim.Adam(model.parameters(), eps=1e-4)

    agent = pfrl.agents.PPO(
        ppo,
        optimizer,
        gpu=-1,
        gamma=0.95,
        # phi=lambda x: x.astype(numpy.float32, copy=False),
        update_interval=2 ** 8,
        minibatch_size=2 ** 6,
        epochs=10,
    )

    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=1000000,  # Train the agent for 2000 steps
        eval_n_steps=None,  # We evaluate for episodes, not time
        eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
        eval_interval=1000,  # Evaluate the agent after every 1000 steps
        train_max_episode_len=100,
        # eval_max_episode_len=10,
        outdir='2020年12月16日',  # Save everything to 'result' directory
    )
