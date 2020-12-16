import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_n, hidden_n, feature_n, output_n):
        super().__init__()

        self.input_n = input_n
        self.feature_n = feature_n
        self.output_n = output_n

        self.fc1 = nn.Linear(self.input_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, feature_n)
        self.fc3 = nn.Linear(feature_n, hidden_n)
        self.fc4 = nn.Linear(hidden_n, output_n)

    def forward(self, input, with_feature=False):
        # [batch, sets, channels, width, height]
        x = input.reshape([input.shape[0] * input.shape[1], self.input_n])
        x = self.fc1(x).relu()
        x = self.fc2(x)
        x = x.reshape([input.shape[0], input.shape[1], self.feature_n])
        x = x.max(1)[0]
        feature = x
        x = self.fc3(x).relu()
        x = self.fc4(x)
        if with_feature:
            return x, feature
        return x


class PPO(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_n):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(obs_size, hidden_n, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_n, hidden_n, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_n, 1, 3, 1, 1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(hidden_n, 1)

    def forward(self, x):
        h = x
        h = self.conv1(h).relu()
        h = self.conv2(h).relu()
        action = self.conv3(h)
        action = action.softmax(1)
        action = action.reshape(x.shape[0], self.n_actions)
        value = self.linear1(self.GAP(h).squeeze(2).squeeze(2))
        return torch.distributions.Categorical(action), value
