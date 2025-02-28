from math import log
import random
from random import seed
from re import A
import re

from requests import get
# # Torch only supports less than 1.0.0 version of gym. We don't want that
# from torchrl.envs.libs.gym import GymEnv as gymenv
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import os
import logging
from collections import defaultdict

#setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed = 42
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed_all(seed)
else:
    device = torch.device("cpu")
    torch.manual_seed(seed)
logger.info(f"Using device: {device}")

# # Set the environment
# env_name = "Blackjack-v1"
# env = gym.make(env_name, natural=False, sab=False, render_mode="human")
# obs, info = env.reset()
# logger.info(f"Environment: {obs}, {info}")

# Blackjack environment
# ACTIONS = [0, 1]  # 0: stick, 1: hit
# NUM_ACTIONS = len(ACTIONS)
# Observation_SPACE = tuple[int, int, int]  # (player_current_sum, dealer_card, usable_ace)
# Start_SPACE = tuple[int, int, int]  # (player_current_sum, dealer_card)
# # Reward SPACE
# win = 0
# lose = 0
# draw = 0
# if win == 1:
#     reward = 1
# elif lose == 1:
#     reward = -1
# elif draw == 1:
#     reward = 0
# else:
#     logger.info("Invalid game")

# Terminal state
# if the sum >= 21, the game is over

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Plot the episode vs total rewards
def plot_episodes_vs_totalrewards(reward_per_episode):
    rolling_length = 500
    reward_moving_average = get_moving_avgs(reward_per_episode, rolling_length, "valid")
    episodes = [i for i in range(len(reward_moving_average))]
    plt.figure() 
    plt.plot(episodes, reward_moving_average)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Episodes vs Total Rewards')
    plt.grid(True)
    plt.show()

class Blackjack_env:   
    def __init__(self, env):
        self.env = env
        self.obs = None
        self.info = None
        # defaultdict where each key maps to a numpy array of zeros 
        # with a length equal to the number of actions in the environment's 
        # action space. This ensures that any state-action pair 
        # that hasn't been encountered yet will have a default Q-value of a zero array.
        self.Q = defaultdict(lambda: np.random.uniform(low=0, high=1, size=env.action_space.n))

    def reset(self):
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    
    def get_state_space(self):
        return self.env.observation_space
    
    def get_action_space(self):
        return self.env.action_space

    def get_obs(self):
        reset_obs = self.env.reset()
        return reset_obs[0]
    
    def get_action(self, current_obs):
        if np.random.uniform(0,1) < 0.1:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[current_obs])

    
if __name__ == "__main__":
    episodes = 2000000
    reward_per_episode = []

    env_name = "Blackjack-v1"
    env = gym.make(env_name, natural=False, sab=False)
    BlackJackagent = Blackjack_env(env=env)
    # state_space = BlackJackagent.get_state_space()
    # action_space = BlackJackagent.get_action_space()
    # logger.info(f"State space: {state_space}, Action space: {action_space}")

    for episode in range(episodes):
        total_reward = 0
        done = False
        env.reset()
        current_obs = BlackJackagent.get_obs()
        while not done:
            action = BlackJackagent.get_action(current_obs)
            obs = current_obs
            #logger.info(f"Episode: {episode}, Action: {action}, Current observation: {current_obs}")
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if reward == 1:
            #     reward = 100
            # elif reward == -1:
            #     reward = -10
            # elif reward == 0:
            #     reward = 0

            # if terminated:
            #     logger.info(f"Episode num: {episode}; terminated:{terminated};current_obs: {current_obs}")

            # #logger.info(f"Episode: {episode}")
            # #logger.info(f"next observation: {next_obs}; QValue: {BlackJackagent.Q[next_obs][:]}; maxQ: {np.max(BlackJackagent.Q[next_obs][:])};")
            
            next_q = (not terminated) * np.max(BlackJackagent.Q[next_obs][:])
            td_error = reward + 0.95 * next_q - BlackJackagent.Q[current_obs][action]
            BlackJackagent.Q[current_obs][action] = BlackJackagent.Q[current_obs][action] + 0.01 * td_error
            #logger.info(f"TD error: {td_error}; QValue: {BlackJackagent.Q[current_obs][action]};")
            total_reward += reward
            if terminated or truncated:
                done = True
                reward_per_episode.append(total_reward)
                break
            else:
                current_obs = next_obs

        if episode % 10000 == 0:
            logger.info(f"Episode: {episode}, Total reward: {total_reward}")

    # Save the Q-table
    logger.info(f"Final Q table: {BlackJackagent.Q}")
    env.close()
    plot_episodes_vs_totalrewards(reward_per_episode)
