'''
Program usage:
    python TD_value_Learning.py --grid_size 20 --alpha 0.01 --gamma 0.9 --episodes 10000 --steps 1000 --loggin --epsilon 0.01
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
from ast import And
from encodings.idna import sace_prefix
from math import log
import re
from turtle import st
import numpy as np
import argparse
import logging
import random
import matplotlib.pyplot as plt

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
        self.GRID = np.zeros((self.N, self.N))
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.goal_state = (self.N-1, self.N-1)
        self.actions = self.get_actions()[0]
        self.V = {state: 0 for state in self.states}
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = {state: {action: (self.epsilon / len(self.actions)) for action in self.actions} for state in self.states}


    def get_best_action(self, state, V, policy):
        actions = self.get_actions()[0]
        best_action = None
        best_value = float('-inf')
        for action in actions:
            next_state = (state[0] + self.get_actions()[1][action][0], state[1] + self.get_actions()[1][action][1])
            if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                pass
            else:
                if V[next_state] > best_value:
                    best_value = V[next_state]
                    best_action = action
        return best_action
    
    def update_policy(self, V, policy):
        for state in self.states:
            actions = self.get_actions()[0]
            best_action = None
            best_value = float('-inf')
            for action in actions:
                next_state = (state[0] + self.get_actions()[1][action][0], state[1] + self.get_actions()[1][action][1])
                if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                    pass
                else:
                    if V[next_state] > best_value:
                        best_value = V[next_state]
                        best_action = action
            for action in actions:
                if action != best_action:
                    policy[state][action] = self.epsilon / (len(actions))
                    #policy[state][action] = 0
                else:
                    policy[state][best_action] = 1 - self.epsilon + (self.epsilon / (len(actions)))
                    #policy[state][best_action] = 1
        return policy
    
    def get_actions(self):
        ACTIONS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        return list(ACTIONS.keys()), ACTIONS 

    def plot_episodes_vs_totalrewards(self, reward_plot_values):
        episodes = [i for i in range(len(reward_plot_values))]
        plt.figure() 
        plt.plot(episodes, reward_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Episodes vs Total Rewards')
        plt.grid(True)
        plt.savefig('images/episode_totalrewards_plot_TD_valuelearn.png')
        plt.show()

    def plot_episodes_vs_steps(self, steps_plot_values):
        episodes = [i for i in range(len(steps_plot_values))]
        plt.figure()
        plt.plot(episodes, steps_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Steps')
        plt.title('Episodes vs Total Steps')
        plt.grid(True)
        plt.savefig('images/episode_steps_plot_TD_valuelearn.png')
        plt.show()

    # Plot the rewards for checking the rewards
    def plot_grid(self,V):
        mV = np.zeros((self.N, self.N))
        for state in self.states:
            mV[state] = V[state]
        plt.figure()
        plt.imshow(mV, cmap='viridis', interpolation='none', origin='lower')
        plt.colorbar()
        #plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.title('2D grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        if self.N < 20:
            plt.xticks(np.arange(0, self.N, 2))
            plt.yticks(np.arange(0, self.N, 2))
        else:
            plt.xticks(np.arange(0, self.N, self.N//5))
            plt.yticks(np.arange(0, self.N, self.N//5))
        plt.savefig('images/grid_plot_TD_valuelearn.png')
        plt.show()

if __name__ == '__main__':
    # Parse the user arguments
    parser = argparse.ArgumentParser(description='TD-Learning for Maze Problem')
    parser.add_argument('--grid_size', type=int, default=20, help='Size of the grid (NxN)')
    parser.add_argument('--alpha', type=restricted_float_alpha, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=restricted_float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=restricted_float, default=0.0001, help='Epsilon for epsilon-greedy policy')
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

    grid = Grid(GRID_SIZE, gamma)
    if loggin:
        logging.info(f"States: {grid.policy}")

    #epsilon_decay = 0.995
        
    for episode in range(episodes):
        # Initialize s
        state = (0, 0)
        total_reward = 0
        for step in range(steps):

            if (random.uniform(0, 1) < epsilon) or (grid.V[state] == 0):
                action = random.choice(list(grid.policy[state].keys()))
            else:
                action = grid.get_best_action(state, grid.V, grid.policy)

            next_state = (state[0] + grid.get_actions()[1][action][0], state[1] + grid.get_actions()[1][action][1])
            if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                next_state = state
                reward = -10
            elif next_state == grid.goal_state:
                reward = 100
            else:
                reward = -1

            TD_error = reward + gamma * grid.V[next_state] - grid.V[state]

            grid.V[state] = grid.V[state] + alpha * TD_error

            total_reward += reward
            
            if next_state == state:
                steps_plot_values.append(step)
                reward_plot_values.append(total_reward)
                break
            elif next_state != state:
                state = next_state
            if state == grid.goal_state:
                steps_plot_values.append(step)
                reward_plot_values.append(total_reward)
                break
        grid.policy = grid.update_policy(grid.V, grid.policy)

        # Decay epsilon
        #epsilon = max(0.01, epsilon * epsilon_decay)

        if episode % 1000 == 0:
            logging.info(f"Episode number: {episode}, Total reward: {total_reward}")
    
    #grid.plot_episodes_vs_totalrewards(reward_plot_values)
    #grid.plot_episodes_vs_steps(steps_plot_values)
    #print(f"Value of V: {grid.V}")
    #print(f"Policy: {grid.policy}")
    grid.plot_grid(grid.V)
