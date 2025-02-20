'''
Program usage:
    python v2_TD_value_learn.py --grid_size 20 --alpha 0.01 --gamma 0.9 --episodes 10000 --steps 1000 --loggin --epsilon 0.01
Problem Description:
    Maze problem using Q-Learning

    1. The agent moves through a maze to find the goal.
    2. The agent receives a reward of 
        a. -1 for each move
        b. -10 for hitting the boundary
        c. +100 for reaching the goal
    3. The agent can move in four directions: up, down, left, and right.
    4. Maze is a 20x20 grid.
    5. The goal is at (19, 19).
    6. The agent starts at (0, 0).
    7. Discount factor gamma is 0.99.
    8. Q-Learning to find the optimal policy.
    9. epsilon-greedy policy with epsilon = 0.01.
    10. The agent uses a learning rate alpha = 0.01
    11. Number of episodes 10000.
    12. Number of steps in each episode is limited to 1000.

'''

import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import logging

# Ensure that matplotlib plots are displayed inline if running in an interactive environment
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# Setup the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x

def restricted_float_alpha(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.1, 0.01]")
    return x

class Grid():
    def __init__(self, N, gamma=0.99, alpha=0.01, epsilon=0.01):
        self.N = N
        self.grid = np.zeros((self.N, self.N))
        self.initial_state = (0, 0)
        self.goal_state = (self.N-1, self.N-1)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon


if __name__ == '__main__':
    # Parse the user arguments
    parser = argparse.ArgumentParser(description='TD-Learning for Maze Problem')
    parser.add_argument('--grid_size', type=int, default=20, help='Size of the grid (NxN)')
    parser.add_argument('--alpha', type=restricted_float_alpha, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=restricted_float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=restricted_float, default=0.01, help='Epsilon for epsilon-greedy policy')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps in each episode')
    parser.add_argument('--loggin', action='store_true', help='Enable logging')
    args = parser.parse_args()

    GRID_SIZE = args.grid_size
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    episodes = args.episodes
    steps = args.steps
    loggin = args.loggin

    if loggin:
        logging.info(f"Grid size: {GRID_SIZE}")
        logging.info(f"Learning rate: {alpha}")
        logging.info(f"Discount factor: {gamma}")
        logging.info(f"Epsilon: {epsilon}")
        logging.info(f"Number of episodes: {episodes}")
        logging.info(f"Number of steps: {steps}")

    # Store (Episodes, total rewards) and (Episodes, total steps)
    reward_plot_values = []
    steps_plot_values = []
