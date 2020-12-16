import random
import gym
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Iterable


def crop(image: Tensor, coord, patch_size: int) -> Tensor:
    assert len(coord) == 2
    assert image.dim() == 3
    # image = nn.ConstantPad2d(patch_size // 2, 0)(image)
    patch = image[
        :,
        coord[0] : coord[0] + patch_size,
        coord[1] : coord[1] + patch_size,
    ]
    assert patch.shape == torch.Size(
        [1, *[patch_size] * 2]
    ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
    return patch


class PatchSetsClassification(gym.Env):
    def __init__(
        self, dataset: Dataset, model: nn.Module, patch_size: int, feature_n: int
    ):
        self.action_space = None
        self.observation_space = None

        for attr in ['encode', 'pool', 'decode']:
            assert hasattr(model, attr), attr

        self.dataset = dataset
        self.model = model
        self.patch_size = patch_size
        self.feature_n = feature_n

        self.patch_list = []
        self.action_list = []

    @staticmethod
    def make_observation(image: Tensor, feature: Tensor) -> Tensor:
        assert image.dim() == 3
        assert feature.dim() == 1
        observation = torch.cat(
            [
                image,
                *feature.squeeze(0)[:, None, None, None].expand(
                    self.feature_n, *image.shape
                ),
            ]
        )
        return observation

    def reset(self, data_index: int = None) -> Tensor:
        with torch.no_grad():
            self.action_list = []
            self.patch_list = []
            self.feature_list = []
            if data_index is None:
                self.data = random.choice(self.dataset)
            else:
                self.data = self.dataset[data_index]
            observation = self.make_observation(
                self.data[0], torch.zeros([self.feature_n])
            )
            self.last_loss = F.binary_cross_entropy_with_logits(
                torch.zeros([1, len(self.data[1])]), self.data[1][None]
            ).item()
        return observation

    def step(self, action: Iterable) -> tuple:
        with torch.no_grad():
            image = self.data[0]

            self.action_list.append(action)
            action_x, action_y = (action // image.shape[1], action % image.shape[2])

            patch = crop(image, [action_x, action_y], self.patch_size)
            self.patch_list.append(patch)

            feature_set = self.model.encode(patch.unsqueeze(0).unsqueeze(0))
            feature_set = torch.cat([feature_set, self.feature_list[-1].squeeze(1)], 1)

            feature = self.model.pool(feature_set)
            self.feature_list.append(feature)

            y_hat = self.model.decode(feature)

            observation = self.make_observation(self.data[0], feature[0])
            loss = F.binary_cross_entropy_with_logits(y, self.data[1][None]).item()
            reward = self.last_loss - loss
            self.last_loss = loss
            done = False
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
    patch_size = 5
    input_n, hidden_n, feature_n, output_n = patch_size ** 2, 16, 16, 2
    obs_size, n_actions = feature_n + 1, image_size ** 2

    model = src.model.Model(input_n, hidden_n, feature_n, output_n)

    dataset = [
        (
            torch.randn(1, image_size, image_size) * 100,
            torch.tensor([1.0] + [0.0] * (output_n - 1)),
        )
        for i in range(1000)
    ]

    env = PatchSetsClassification(dataset, model, patch_size, feature_n)

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
        gamma=0.99,
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
        steps=10000,  # Train the agent for 2000 steps
        eval_n_steps=None,  # We evaluate for episodes, not time
        eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
        eval_interval=1000,  # Evaluate the agent after every 1000 steps
        train_max_episode_len=32,
        # eval_max_episode_len=10,
        outdir='patch',  # Save everything to 'result' directory
    )
