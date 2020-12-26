import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union

from src.unet import UNet


# class Policy(nn.Module):
#     def __init__(self, obs_size, patch_size):
#         super().__init__()
#         self.obs_size = obs_size

#         self.unet = UNet(n_channels=obs_size, n_classes=None)
#         self.conv = nn.Conv2d(64, 1, patch_size, 1, 0)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.unet(x)
#         x = self.conv(x)
#         x = self.flatten(x)
#         x = x.softmax(1)
#         x = torch.distributions.Categorical(probs=x)
#         return x


# class Value(nn.Module):
#     def __init__(self, obs_size, hidden_n):
#         super().__init__()
#         self.obs_size = obs_size

#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(obs_size, hidden_n, 3, 1, 1)
#         self.conv2 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
#         self.conv3 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
#         self.conv4 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
#         self.flaten = nn.Flatten()
#         self.linear = nn.Linear(hidden_n * (100 // (2 ** 2)) ** 2, 1)

#     def forward(self, x):
#         x = self.conv1(x).relu()
#         x = self.pool(x)
#         x = self.conv2(x).relu()
#         x = self.conv3(x).relu()
#         x = self.pool(x)
#         x = self.conv4(x).relu()
#         x = self.flaten(x)
#         x = self.linear(x)
#         return x


# class Model(nn.Module):
#     def __init__(self, obs_size, n_actions, patch_size):
#         super().__init__()
#         self.obs_size = obs_size
#         self.n_actions = n_actions

#         self.policy = Policy(obs_size, patch_size)
#         self.value = Value(obs_size, 64)

#     def forward(self, x):
#         action = self.policy(x)
#         value = self.value(x)
#         return action, value


class Model(nn.Module):
    def __init__(self, obs_size, patch_size):
        super().__init__()
        self.obs_size = obs_size

        self.unet = UNet(n_channels=obs_size, n_classes=None)
        self.action_conv = nn.Conv2d(64, 1, patch_size, 1, 0)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.policy_fc = nn.Linear(512, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        action, policy = self.unet(x, return_feature=True)

        action = self.action_conv(action)
        action = self.flatten(action)
        action = action.softmax(1)
        action = torch.distributions.Categorical(probs=action)

        policy = self.gap(policy)
        policy = self.flatten(policy)
        policy = self.policy_fc(policy)

        return action, policy


if __name__ == "__main__":
    from torchsummary import summary

    model = Model(obs_size=2, patch_size=25)
    summary(model, [2, 2, 100, 100])
