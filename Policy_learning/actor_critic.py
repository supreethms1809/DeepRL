import os
from venv import logger
import numpy as np
import argparse
import logging
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = 'cuda'
    logging.info("CUDA is available. Running on Nvidia GPU")
elif torch.mps.is_available():
    device = 'mps'
    logging.info("MPS is available. Running on Apple M4 GPU")
else:
    device = 'cpu'
    logging.info("CUDA and MPS are not available. Running on CPU")

# Racing environment
class RacingEnv(nn.Module):
    def __init__(self, env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=100):
        nn.Module.__init__(self)
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        self.Q = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 48, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
            ).to(device)

        self.Qoptimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.Qloss = nn.MSELoss()
        self.targetQ = self.Q

    def forward(self, x):
        return self.Q(x)
    
    def updatetargetQ(self, Q):
        self.targetQ = Q
        return self.targetQ
    
# Define Policy network
class RacingPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.policy = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 48, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
            ).to(device)

        self.policyoptimizer = optim.Adam(self.parameters(), lr=alpha)
        self.policyloss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.policy(x).softmax(dim=1)
    
# Define Agent
class RacingAgent(RacingEnv, RacingPolicy):
    def __init__(self, env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=100, batch_size=32):
        RacingEnv.__init__(self, env, input_dim, output_dim, epsilon, gamma, alpha, episodes)
        RacingPolicy.__init__(self, input_dim, output_dim, alpha)
        self.batch_size = batch_size
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy(state).argmax().item()

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            state = torch.tensor(state[0], dtype=torch.float32).to(device)
            state = torch.permute(state, (2, 1, 0))
            state = state.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
            done = False
            logger.info(f"Policy: {self.policy}")
            logger.info(f"Q: {self.Q}")
            # Train the Q network
            while not done:
                action = self.get_action(state)
                logger.info(f"Action: {action}")
                done = True




if __name__ == "__main__":

    epsilon = 0.01
    gamma = 0.9
    alpha = 0.01
    episodes = 100
    batch_size = 32

    env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode=None)
    input_dim = env.observation_space.shape[2]
    output_dim = env.action_space.n
    logger.info(f"Input channels: {input_dim}, Output dimension: {output_dim}")
    agent = RacingAgent(env, input_dim, output_dim, epsilon, gamma, alpha, episodes, batch_size)
    agent.to(device)
    agent.train()