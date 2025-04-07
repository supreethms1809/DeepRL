from venv import logger
import torch
import trl
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import os
import logging
import sys

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
# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
elif device == 'mps':
    torch.mps.manual_seed_all(42)
else:
    torch.manual_seed(42)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

# Load the dataset
ds = load_dataset("open-r1/codeforces")

class PPOAgent():
    def learn(self):
        pass

class ActorCritic(nn.Module):
    def __init__(self,output_dim):
        super(ActorCritic, self).__init__()
        # This is my base model - policy
        self.actor = nn.Sequential(
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=768, nhead=8), num_layers=6),
            nn.Linear(768, output_dim)
        )
        self.critic = nn.Sequential(
            nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=768, nhead=8), num_layers=6),
            nn.Linear(768, 1)
        )
    
    def forward(self, x):
        actor_out = self.actor(x)
        critic_out = self.critic(x)
        return actor_out, critic_out
    
ac = ActorCritic()
logger.info(f"ActorCritic model: {ac}")


       