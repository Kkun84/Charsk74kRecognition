import pfrl
import torch
from torch import nn
import gym
import numpy
import pytorch_lightning as pl


# pl.seed_everything(0)

env = gym.make("CartPole-v0")
print("observation space:", env.observation_space)
print("action space:", env.action_space)

obs = env.reset()
print("initial observation:", obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print("next observation:", obs)
print("reward:", r)
print("done:", done)
print("info:", info)


# class PPO(nn.Module):
#     def __init__(self, obs_size, n_actions):
#         super().__init__()
#         self.input_shape = [obs_size]
#         self.output_shape = [n_actions]

#         self.linear1 = nn.Linear(obs_size, 2 ** 5)
#         self.linear2 = nn.Linear(2 ** 5, 2 ** 5)
#         self.linear3 = nn.Linear(2 ** 5, n_actions)
#         self.linear4 = nn.Linear(2 ** 5, 1)

#     def forward(self, x):
#         h = x
#         h = self.linear1(h).relu()
#         h = self.linear2(h).relu()
#         action = self.linear3(h).softmax(1)
#         value = self.linear4(h)
#         return torch.distributions.Categorical(action), value


class PPO(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.input_shape = [obs_size]
        self.output_shape = [n_actions]

        self.linear1 = nn.Linear(obs_size, 2 ** 5)
        self.linear2 = nn.Linear(2 ** 5, 2 ** 5)
        self.linear3 = nn.Linear(2 ** 5, n_actions)
        self.linear4 = nn.Linear(obs_size, 2 ** 5)
        self.linear5 = nn.Linear(2 ** 5, 2 ** 5)
        self.linear6 = nn.Linear(2 ** 5, 1)

    def forward(self, x):
        h = x
        h = self.linear1(h).relu()
        h = self.linear2(h).relu()
        action = self.linear3(h).softmax(1)
        h = x
        h = self.linear4(h).relu()
        h = self.linear5(h).relu()
        value = self.linear6(h)
        return torch.distributions.Categorical(action), value


obs_size = env.observation_space.low.size
n_actions = env.action_space.n
model = PPO(obs_size, n_actions)


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)


# Now create an agent that will interact with the environment.
agent = pfrl.agents.PPO(
    model,
    optimizer,
    gpu=0,
    gamma=0.9,
    phi=lambda x: x.astype(numpy.float32, copy=False),
    update_interval=2 ** 10,
    minibatch_size=2 ** 6,
    epochs=10,
)


# Set up the logger to print info messages for understandability.
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=100000,  # Train the agent for 2000 steps
    eval_n_steps=None,  # We evaluate for episodes, not time
    eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
    eval_interval=1000,  # Evaluate the agent after every 1000 steps
    outdir='log/ppo/'
    + datetime.now().strftime(
        '%Y-%m-%d-%H-%M-%S'
    ),  # Save everything to 'result' directory
    use_tensorboard=True,
)
