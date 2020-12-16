import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union


class Model(nn.Module):
    def __init__(self, input_n, hidden_n, feature_n, output_n):
        super().__init__()

        self.input_n = input_n
        self.feature_n = feature_n
        self.output_n = output_n

        self.encode_f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_n, hidden_n),
            nn.Tanh(),
            nn.Linear(hidden_n, feature_n),
        )
        self.decode_f = nn.Sequential(
            nn.Linear(feature_n, hidden_n),
            nn.Tanh(),
            nn.Linear(hidden_n, output_n),
        )

    def encode(
        self, patch_set: Union[Iterable[Tensor], Tensor]
    ) -> Union[Iterable[Tensor], Tensor]:
        if isinstance(patch_set, Tensor):
            x = patch_set.reshape(
                patch_set.shape[0] * patch_set.shape[1], *patch_set.shape[2:]
            )
            x = self.encode_f(x)
            x = x.reshape(patch_set.shape[0], patch_set.shape[1], self.feature_n)
        else:
            x = [self.encode_f(x) for x in x]
        return x

    def pool(self, feature_set: Union[Iterable[Tensor], Tensor]) -> Tensor:
        if isinstance(feature_set, Tensor):
            feature = feature_set.sum(1)
        else:
            feature = torch.stack([x.sum(0) for x in feature_set])
        return feature

    def decode(self, feature: Tensor) -> Tensor:
        x = self.decode_f(feature)
        return x * 10

    def forward(self, patch_set, with_feature=False):
        # [batch, sets, channels, width, height]
        feature_set = self.encode(patch_set)
        feature = self.pool(feature_set)
        x = self.decode(feature)
        return x


class PPO(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_n):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(obs_size, hidden_n, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.conv4 = nn.Conv2d(hidden_n, 1, 3, 1, 1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(hidden_n, 1)

    def forward(self, x):
        h = x
        h = self.conv1(h).relu()
        h = self.conv2(h).relu()
        h = self.conv3(h).relu()

        action = self.conv4(h)
        action = action.softmax(1)
        action = action.reshape(x.shape[0], self.n_actions)

        value = self.linear1(self.GAP(h).squeeze(2).squeeze(2))

        return torch.distributions.Categorical(action), value
