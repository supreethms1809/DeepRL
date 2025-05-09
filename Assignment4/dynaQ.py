import os
import re
from unittest import result
import numpy as np
import random
import argparse
import logging
import matplotlib.pyplot as plt
import json
import datetime
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x

class Grid():
    def __init__(self, N, gamma=0.99, alpha=0.01, epsilon=0.01, planning_steps=4):
        self.N = N
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.goal_state = (self.N - 1, self.N - 1)  # (19,19) for 20x20 grid
        self.start_state = (0, 0)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Actions: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
        # These could've been defined wrong in the previous case. Maybe I got confused with the order especially wrt numpy.
        self.ACTIONS = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Q-table: dictionary of dictionaries
        self.Q = {state: {action: 0 for action in self.ACTIONS.keys()} for state in self.states}

        # Dyna-Q parameters
        self.planning_steps = planning_steps
        self.dynaQ_storage = {}

    def get_actions(self):
        return list(self.ACTIONS.keys())

    # Epsilon-greedy action selection. Previosly, this was defined differently. Lets use this version.
    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            return max(self.Q[state], key=self.Q[state].get)
        
    def model_q(self, state, done=False):
        action = self.epsilon_greedy(state)
        dx, dy = self.ACTIONS[action]
        next_state = (state[0] + dx, state[1] + dy)
        if not (0 <= next_state[0] < self.N and 0 <= next_state[1] < self.N):
            reward = -10
            next_state = state
            done = True
        elif next_state == self.goal_state:
            reward = 100
            next_state = self.goal_state
            done = True
        else:
            reward = -1
        return next_state, reward, action, done
    
    def model(self, state, action):
        dx, dy = self.ACTIONS[action]
        next_state = (state[0] + dx, state[1] + dy)
        if not (0 <= next_state[0] < self.N and 0 <= next_state[1] < self.N):
            reward = -10
            next_state = state
        elif next_state == self.goal_state:
            reward = 100
            next_state = self.goal_state
        else:
            reward = -1
        return reward, next_state
    
    def updateQ(self, state, action, reward, next_state):
        best_next_q = max(self.Q[next_state].values())
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        
    def plot_rewards(self, rewards):
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episodes vs Total Rewards (Q-Learning)")
        plt.grid(True)
        dt = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        plt.savefig('dt'+'qlearning_rewards.png')
        plt.show()

    def plot_steps(self, steps):
        plt.figure()
        plt.plot(steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps to Goal")
        plt.title("Episodes vs Steps to Goal (Q-Learning)")
        plt.grid(True)
        dt = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        plt.savefig('dt'+'qlearning_steps.png')
        plt.show()

    def plot_grid(self):
        value_grid = np.zeros((self.N, self.N))
        for state in self.states:
            value_grid[state] = max(self.Q[state].values())
        plt.figure()
        plt.imshow(value_grid, cmap='jet', origin='lower')
        plt.colorbar(label='Max Q-value')
        plt.title('Learned Q Value Table')
        plt.xlabel('X')
        plt.ylabel('Y')
        dt = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        plt.savefig('dt'+'qlearning_value_table.png')
        plt.show()

    def plot_moving_average(self, steps):
        episodes = np.arange(len(steps))
        window_size = 100
        moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        moving_std = np.array([
            np.std(steps[max(0, i - window_size):i])
            for i in range(window_size, len(steps) + 1)
        ])
        episodes_smooth = episodes[window_size - 1:]
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, steps, alpha=0.3, label='Raw Steps')
        plt.plot(episodes_smooth, moving_avg, color='orange', label='Moving Average')
        plt.fill_between(episodes_smooth,
                        moving_avg - moving_std,
                        moving_avg + moving_std,
                        color='orange',
                        alpha=0.3,
                        label='±1 Std Dev')

        plt.title("Episodes vs Steps to Goal (Q-Learning)")
        plt.xlabel("Episode")
        plt.ylabel("Steps to Goal")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        dt = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        plt.savefig('dt'+'qlearning_moving_average.png')
        plt.show()

    def plt_moving_avgs(self, steps):
        episodes = np.arange(len(steps))
        coefficients = np.polyfit(episodes, steps, 6)
        trend_line = np.polyval(coefficients, episodes)
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, steps, label='Raw Steps', alpha=0.5)
        plt.plot(episodes, trend_line, label='Trend Line', color='red', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Steps to Goal with Trend Line')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        dt = f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        plt.savefig('dt'+'qlearning_trend_line.png')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning for Maze Navigation")
    parser.add_argument('--grid_size', type=int, default=20, help='Size of the grid (NxN)')
    parser.add_argument('--alpha', type=restricted_float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=restricted_float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=restricted_float, default=0.01, help='Epsilon for exploration')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--planning_steps', type=int, default=0, help='Number of planning steps in Dyna-Q')
    parser.add_argument('--loggin', action='store_true', help='Enable logging')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    args = parser.parse_args()

    if args.planning_steps > 0:
        logging.info("Dyna-Q with planning steps enabled.")
        dynaQ = True
    else:
        logging.info("Dyna-Q without planning steps.")
        dynaQ = False

    # Initialize environment. I changed few things here to be succint.
    grid = Grid(args.grid_size, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon, planning_steps=args.planning_steps)
    reward_history = []
    step_history = []

    # Run DynaQ-learning algorithm
    for episode in range(args.episodes):
        state = grid.start_state
        total_reward = 0
        done = False

        for step in range(args.steps):
            # Q-learning components 
            next_state, reward, action, done = grid.model_q(state, done=False)
            # Update Q-value
            grid.updateQ(state, action, reward, next_state)

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # End episode if done
            if done:
                break

            # Step e from the psuedo code.
            if dynaQ:
                # Dyna-Q update (planning) Store the experience in the model
                grid.dynaQ_storage[(state,action)] = (reward, next_state)
        
        # Step f from the psuedo code. planning stage
        if dynaQ:
            # Dyna-Q planning steps
            for _ in range(grid.planning_steps):
                # sample a random experience from the storage, run the model to
                # get the next state and reward, and update Q-value
                if len(grid.dynaQ_storage) > 0:
                    sampled_data = random.choice(list(grid.dynaQ_storage.items()))
                    key, value = sampled_data
                    sampled_state, sampled_action = key[0], key[1]
                    sampled_reward, sampled_next_state = grid.model(sampled_state, sampled_action)
                    # Update Q-value using the sampled experience
                    grid.updateQ(sampled_state, sampled_action, sampled_reward, sampled_next_state)

        reward_history.append(total_reward)
        step_history.append(step)

        if args.loggin and episode % 10000 == 0:
            logging.info(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step}")

    #Save reward history, step history, and Q-table 
    results = {
        'planning_steps': args.planning_steps,
        'reward_history': reward_history,
        'step_history': step_history,
        'Q_table': {str(key): value for key, value in grid.Q.items()}
    }
    
    file_path = "results.json"

    # Check if the file exists
    if os.path.exists(file_path):
        # Read existing data
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(results)

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Data appended to {file_path}")

    # Plot results
    if args.plot:
        grid.plot_rewards(reward_history)
        grid.plot_steps(step_history)
        grid.plot_moving_average(step_history)
        grid.plt_moving_avgs(step_history)
        grid.plot_grid()