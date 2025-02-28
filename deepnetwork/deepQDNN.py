import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader 
import gymnasium as gym
import random
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        self.denseQ = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.denseQ_target = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.optimizer = AdamW(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.denseQ(x)
        return x
    
    def updateQTarget(self):
        self.denseQ_target = self.denseQ
        return self.denseQ_target
    
class RQN(nn.Module):
    def __init__(self, input_size, output_size):
        self.densePolicy = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.optimizer = AdamW(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.densePolicy(x)
        return x
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, state, action, next_state, reward):
        pass

    def get_batch(self, batch_size):
        pass

class DQNAgent:
    def __init__(self):
        pass

    # def get_batch(self, batch_size):
    #     pass

    def computeTarget(self):
        pass

    def updateQweights(self):
        pass

if __name__ == "__main__":
    # Starting RL training 
    logger.info("Starting RL training")

    env_name = "CarRacing-v3"
    env = gym.make(env_name, lap_complete_percent=0.95)
    #env = gym.make(env_name, render_mode="human")

    obs, info = env.reset()
    logger.info(f"Observation space: {env.observation_space}, Action space: {env.action_space}")