import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union

from src.unet import UNet


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
