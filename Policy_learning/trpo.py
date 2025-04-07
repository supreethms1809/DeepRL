from json import load
from operator import ne
import os
from venv import logger
import numpy as np
import argparse
import logging
import sys
import logging
from sympy import Q
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
from tqdm import tqdm

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
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 48 * 48, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
            ).to(device)

        self.Qoptimizer = optim.AdamW(self.parameters(), lr=self.alpha, weight_decay=1e-2, betas=(0.9, 0.99))
        self.Qloss = nn.MSELoss()
        self.targetQ = self.Q

    def forward(self, x):
        return self.Q(x)
    
    def updatetargetQ(self, source_network):
        """
        Copy the weights from source_network to targetQ.
        """
        self.targetQ.load_state_dict(source_network.state_dict())
        return self.targetQ
    
    def get_Q_target_state_action(self, state, action):
        y = self.targetQ(state).gather(1, action.unsqueeze(1))
        #logger.info(f"Target Q: {y.shape} and action = {action}")
        return y
    
    def get_Q_state_action(self, state, action):
        y = self.Q(state).gather(1, action.unsqueeze(1))
        #logger.info(f"Q: {y.shape} and action = {action}")
        return y
    
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
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 48 * 48, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
            ).to(device)

        self.policyoptimizer = optim.AdamW(self.parameters(), lr=self.alpha, weight_decay=1e-2, betas=(0.9, 0.99))
        #self.policyloss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.policy(x).softmax(dim=1)
    
# Define Agent
class RacingAgent(RacingEnv, RacingPolicy):
    def __init__(self, env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=100, batch_size=8):
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
            t_action = torch.tensor([self.env.action_space.sample() for _ in range(state.shape[0])], device=device)
        else:
            #logger.info(f"Policy: {self.policy(state).argmax(dim=1)}")
            with torch.no_grad():
                t_action = self.policy(state).argmax(dim=1)
        return t_action
    
    def collect_trajectory(self, episode, initial_state):
        """
        Collect a trajectory of data from the environment.
        """
        state = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
        done_traj = False
        trajectory = []
        
        while not done_traj:
            action = self.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
            trajectory.append((episode, state, action, reward, next_state))
            state = next_state.clone().detach()
            if done or truncated:
                done_traj = True
        
        return trajectory


if __name__ == "__main__":
    episodes = 2
    env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode=None)
    input_dim = env.observation_space.shape[2]
    output_dim = env.action_space.n
    agent = RacingAgent(env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=episodes)

    for episode in tqdm(range(episodes)):
        # Reset the environment for each episode
        state, info = env.reset()
        trajectory = agent.collect_trajectory(episode, state)
        
        batch_size = 8

        for batch in range(0, len(trajectory), batch_size):
            batch_trajectory = trajectory[batch:batch + batch_size]
            if len(batch_trajectory) < batch_size:
                break
            states = torch.stack([t[1] for t in batch_trajectory]).to(device)
            actions = torch.stack([t[2] for t in batch_trajectory]).to(device)
            rewards = torch.tensor([t[3] for t in batch_trajectory], dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.stack([t[4] for t in batch_trajectory]).to(device)
            states = states.squeeze()
            actions = actions.squeeze()
            rewards = rewards.squeeze()
            next_states = next_states.squeeze()
            logger.info(f"States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Next States: {next_states.shape}")

            # calculate TD_target
            current_Q = agent.get_Q_state_action(states, actions)
            next_Q = agent.get_Q_target_state_action(next_states, actions)
            TD_target = rewards + agent.gamma * next_Q
            advantage_function = TD_target - current_Q

            


            
        