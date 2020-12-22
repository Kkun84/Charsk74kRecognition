import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union

from .unet import UNet


# class Policy(nn.Module):
#     def __init__(self, obs_size, n_actions, hidden_n, patch_size):
#         super().__init__()
#         self.obs_size = obs_size
#         self.n_actions = n_actions

#         self.conv1 = nn.Conv2d(obs_size, hidden_n, 5, 1, 2)
#         self.conv2 = nn.Conv2d(hidden_n, hidden_n, 5, 1, 2)
#         self.conv3 = nn.Conv2d(hidden_n, hidden_n, 5, 1, 2)
#         self.conv4 = nn.Conv2d(hidden_n, hidden_n, 5, 1, 2)
#         self.conv5 = nn.Conv2d(hidden_n, hidden_n, 5, 1, 2)
#         self.conv6 = nn.Conv2d(hidden_n, 1, patch_size, 1, 0)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.conv1(x).relu()
#         x = self.conv2(x).relu()
#         x = self.conv3(x).relu()
#         x = self.conv4(x).relu()
#         x = self.conv5(x).relu()
#         x = self.conv6(x)
#         x = x.softmax(1)
#         x = self.flatten(x)
#         return torch.distributions.Categorical(x)


class Policy(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_n, patch_size):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.unet = UNet(n_channels=obs_size, n_classes=None)
        self.conv = nn.Conv2d(64, 1, patch_size, 1, 0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.unet(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = x.softmax(1)
        x = torch.distributions.Categorical(probs=x)
        return x


class Value(nn.Module):
    def __init__(self, obs_size, hidden_n):
        super().__init__()
        self.obs_size = obs_size

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(obs_size, hidden_n, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.conv4 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.flaten = nn.Flatten()
        self.linear = nn.Linear(hidden_n * (100 // (2 ** 2)) ** 2, 1)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.pool(x)
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.pool(x)
        x = self.conv4(x).relu()
        x = self.flaten(x)
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_n, patch_size):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.policy = Policy(obs_size, n_actions, hidden_n, patch_size)
        self.value = Value(obs_size, hidden_n)

    def forward(self, x):
        action = self.policy(x)
        value = self.value(x)
        return action, value


# class PPO(nn.Module):
#     def __init__(self, obs_size, n_actions, hidden_n, patch_size):
#         super().__init__()
#         self.obs_size = obs_size
#         self.n_actions = n_actions

#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(100 ** 2, 86 ** 2)
#         self.linear2 = nn.Linear(65 * 100 ** 2, 1)
#         self.policy = pfrl.policies.SoftmaxCategoricalHead()

#     def forward(self, x):
#         x = self.flatten(x)
#         action = self.linear1(x[:, : 100 ** 2])
#         action = self.policy(action)
#         value = self.linear2(x)
#         return action, value


if __name__ == "__main__":
    from torchsummary import summary

    summary(Model(1, 86 ** 2, 64, 15))
