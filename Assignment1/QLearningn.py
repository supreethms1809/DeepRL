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
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

# Setup the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check the input arguments
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

#Define the maze grid
class Grid():
    def __init__(self, N, gamma=0.9, alpha=0.01, epsilon=0.01):
        self.N = N
        self.GRID = np.zeros((self.N, self.N))
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.goal_state = (self.N-1, self.N-1)
        self.Q = {state: {action: 0 for action in self.get_actions()[0]} for state in self.states}
        self.policy = {state: random.choice(self.get_actions()[0]) for state in self.states}
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    # Define the actions for the agent to move in the maze
    def get_actions(self):
        ACTIONS = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        return list(ACTIONS.keys()), ACTIONS 
    
    # Define the epsilon-greedy policy based on the equation. 
    # pi_dash is the probability distribution for all the actions and then we adjust Pob for max action 
    def epsilon_greedy(self, state, actions):
        max_action = np.argmax([self.Q[state][action] for action in actions])
        pi_dash = [self.epsilon/len(actions) for _ in range(len(actions))]
        pi_dash[max_action] += 1 - self.epsilon

        action = np.random.choice(actions, p=pi_dash)
        return action

    # Update the policy based on the Q-values for printing the optimal policy.
    def update_policy(self, state, action):
        self.policy[state] = action

    # Plot the episode vs total rewards
    def plot_episodes_vs_totalrewards(self, reward_plot_values):
        episodes = [i for i in range(len(reward_plot_values))]
        plt.figure() 
        plt.plot(episodes, reward_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Episodes vs Total Rewards')
        plt.grid(True)
        plt.savefig('images/episode_totalrewards_plot_Qlearn.png')
        plt.show()

    # Plot the stepa for each episode
    def plot_episodes_vs_steps(self, steps_plot_values):
        episodes = [i for i in range(len(steps_plot_values))]
        plt.figure()
        plt.plot(episodes, steps_plot_values)
        plt.xlabel('Episodes')
        plt.ylabel('Total Steps')
        plt.title('Episodes vs Total Steps')
        plt.grid(True)
        plt.savefig('images/episode_steps_plot_Qlearn.png')
        plt.show()

    # Plot the max Q-values for each state
    def plot_grid(self,Q):
        maxQ = np.zeros((self.N, self.N))
        for state in self.states:
            maxQ[state] = max(Q[state].values())
        plt.figure()
        plt.imshow(maxQ, cmap='viridis', interpolation='none', origin='lower')
        plt.colorbar()
        #plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.title('Maximum Q-values for each state')
        plt.xlabel('X')
        plt.ylabel('Y')
        if self.N < 20:
            plt.xticks(np.arange(0, self.N, 2))
            plt.yticks(np.arange(0, self.N, 2))
        else:
            plt.xticks(np.arange(0, self.N, self.N//5))
            plt.yticks(np.arange(0, self.N, self.N//5))
        plt.savefig('images/grid_plot_Qlearn.png')
        plt.show()

if __name__ == '__main__':
    # Parse the user arguments
    parser = argparse.ArgumentParser(description='Q-Learning for Maze Problem')
    parser.add_argument('--grid_size', type=int, default=20, help='Size of the grid (NxN)')
    parser.add_argument('--alpha', type=restricted_float_alpha, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=restricted_float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=restricted_float, default=0.01, help='Epsilon for epsilon-greedy policy')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps in each episode')
    parser.add_argument('--loggin', action='store_true', help='Enable logging')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    args = parser.parse_args()

    GRID_SIZE = args.grid_size
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    episodes = args.episodes
    steps = args.steps
    loggin = args.loggin
    plot = args.plot

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

    # Initialize the grid
    grid = Grid(GRID_SIZE, gamma=gamma, alpha=alpha , epsilon=epsilon)
    if loggin:
        logging.info(f"Rewards: {grid.rewards}")
        
    # Q-Learning
    for episode in range(episodes):
        # Initialize s
        state = (0, 0)
        total_reward = 0
        hitBoundary = False
        for step in range(steps):
            # Choose a from s using policy derived from Q
            action = grid.epsilon_greedy(state, grid.get_actions()[0])
            # Take action a, calc s'
            next_state = (state[0] + grid.get_actions()[1][action][0], state[1] + grid.get_actions()[1][action][1])
            if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                hitBoundary = True
                next_state = state
                reward = -10
            elif next_state == grid.goal_state:
                reward = 100
            else:
                reward = -1

            if state != grid.goal_state:
                maxQ = max(grid.Q[next_state].values())
            else:
                maxQ = 0
                reward = 0
            
            # Q update
            grid.Q[state][action] = grid.Q[state][action] + alpha * (reward + gamma * maxQ - grid.Q[state][action])

            # Update policy to print the optimal policy. The updated policy is not used directly here.
            grid.update_policy(state, action)
            total_reward += reward
            state = next_state

            # Check if the agent hit boundary or reached the goal
            if hitBoundary == True:
                steps_plot_values.append(step)
                reward_plot_values.append(total_reward)
                break
            if state == grid.goal_state:
                # Set the Q-values for the goal state to 100. This is not required for the Q-learning algorithm. 
                # if set, the Q-values for the other states go over 100. 
                # This method converges faster compared to not updating Q-values for the goal state.
                # adjust the rewards

                # for action in grid.get_actions()[0]:
                #     grid.Q[grid.goal_state][action] = 100
                steps_plot_values.append(step)
                reward_plot_values.append(total_reward)
                break
        if episode % 1000 == 0:
            logging.info(f"Episode number: {episode}, Total reward: {total_reward}")
    
    if plot == True:
        grid.plot_episodes_vs_totalrewards(reward_plot_values)
        grid.plot_episodes_vs_steps(steps_plot_values)
        grid.plot_grid(grid.Q)




