from logging import getLogger

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union, Dict

from src.unet import UNet


logger = getLogger(__name__)


class AgentModel(nn.Module):
    def __init__(self, input_n: int, patch_size: Dict[str, int]):
        super().__init__()
        self.input_n = input_n

        self.unet = UNet(n_channels=input_n, n_classes=None)
        self.action_conv = nn.Conv2d(64, 1, (patch_size['y'], patch_size['x']), 1, 0)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.policy_fc = nn.Linear(512, 1)

        self.flatten = nn.Flatten()

    def forward(self, x, deterministic_action=False):
        action, policy = self.unet(x, return_feature=True)

        action = self.action_conv(action)
        action = self.flatten(action)
        action = torch.cat([action, torch.zeros_like(action[:, :1])], 1)
        action = action.softmax(1)
        if deterministic_action == True:
            action = (action == action.max(1, True)[0]).float()
        action = torch.distributions.Categorical(probs=action)

        policy = self.gap(policy)
        policy = self.flatten(policy)
        policy = self.policy_fc(policy)

        return action, policy


if __name__ == "__main__":
    from torchsummary import summary

    model = AgentModel(input_n=2, patch_size={'x': 12, 'y': 23})
    summary(model)
