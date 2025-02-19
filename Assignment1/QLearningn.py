'''
Program usage:
    python QLearning.py --grid_size 20 --alpha 0.01 --gamma 0.9 --epsilon 0.01 --episodes 10000 --steps 1000 --loggin

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
    7. Discount factor gamma is 0.9.
    8. Q-Learning to find the optimal policy.
    9. epsilon-greedy policy with epsilon = 0.01.
    10. The agent uses a learning rate alpha = 0.01
    11. Number of episodes 10000.
    12. Number of steps in each episode is limited to 1000.

'''
from math import log
from turtle import st
import numpy as np
import argparse
import logging
import random
import matplotlib.pyplot as plt

# Ensure that matplotlib plots are displayed inline if running in an interactive environment
try:
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
    def __init__(self, N, gamma=0.9, alpha=0.01, epsilon=0.01):
        self.N = N
        self.GRID = np.zeros((self.N, self.N))
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.goal_state = (self.N-1, self.N-1)
        self.rewards = self.set_rewards()
        self.Q = {state: {action: 0 for action in self.get_actions()[0]} for state in self.states}
        self.policy = {state: random.choice(self.get_actions()[0]) for state in self.states}
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    # Define the rewards for each state-action pair
    def set_rewards(self):
        actions, ACTIONS = self.get_actions()
        rewards = {}
        for state in self.states:
            for action in actions:
                if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= self.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= self.N):
                    next_state = state
                    rewards[(state, action)] = -10
                else:
                    next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                    rewards[(state, action)] = -1
                if next_state == self.goal_state:
                    rewards[(state, action)] = 100
        return rewards

    def get_actions(self):
        ACTIONS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        return list(ACTIONS.keys()), ACTIONS 
    
    def epsilon_greedy(self, state, actions):
        max_action = np.argmax([self.Q[state][action] for action in actions])
        pi_dash = [self.epsilon/len(actions) for _ in range(len(actions))]
        pi_dash[max_action] += 1 - self.epsilon

        action = np.random.choice(actions, p=pi_dash)
        return action

    def update_policy(self, state, action):
        self.policy[state] = action

    def plot_episodes_vs_totalrewards(self, reward_plot_values):
        episodes = [i for i in range(len(reward_plot_values))]
        plt.figure() 
        plt.plot(episodes, reward_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Episodes vs Total Rewards')
        plt.grid(True)
        plt.savefig('images/episode_totalrewards_plot_Qlearn.png')
        #plt.show()

    def plot_episodes_vs_steps(self, steps_plot_values):
        episodes = [i for i in range(len(steps_plot_values))]
        plt.figure()
        plt.plot(episodes, steps_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Steps')
        plt.title('Episodes vs Total Steps')
        plt.grid(True)
        plt.savefig('images/episode_steps_plot_Qlearn.png')
        #plt.show()

    # Plot the rewards for checking the rewards
    def plot_grid(self,Q):
        maxQ = np.zeros((self.N, self.N))
        for state in self.states:
            maxQ[state] = max(Q[state].values())
        plt.figure()
        plt.imshow(maxQ, cmap='viridis', interpolation='none', origin='lower', vmin=-100, vmax=100)
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
        plt.savefig('images/grid_plot_Qlearn.png')
        #plt.show()

if __name__ == '__main__':
    # Parse the user arguments
    parser = argparse.ArgumentParser(description='Q-Learning for Maze Problem')
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

    grid = Grid(GRID_SIZE, gamma)
    if loggin:
        logging.info(f"Rewards: {grid.rewards}")
        
    for episode in range(episodes):
        # Initialize s
        state = (0, 0)
        total_reward = 0
        for step in range(steps):
            action = grid.epsilon_greedy(state, grid.get_actions()[0])
            next_state = (state[0] + grid.get_actions()[1][action][0], state[1] + grid.get_actions()[1][action][1])
            if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                next_state = state
            reward = grid.rewards[(state, action)]
            grid.Q[state][action] = grid.Q[state][action] + alpha * (reward + gamma * max(grid.Q[next_state].values()) - grid.Q[state][action])
            grid.update_policy(state, action)
            total_reward += reward
            state = next_state
            if state == grid.goal_state:
                steps_plot_values.append(step)
                reward_plot_values.append(total_reward)
                break
        if episode % 1000 == 0:
            logging.info(f"Episode number: {episode}, Total reward: {total_reward}")
    
    grid.plot_episodes_vs_totalrewards(reward_plot_values)
    grid.plot_episodes_vs_steps(steps_plot_values)
    grid.plot_grid(grid.Q)




