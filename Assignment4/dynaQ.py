import numpy as np
import random
import argparse
import logging
import matplotlib.pyplot as plt

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
    def __init__(self, N, gamma=0.99, alpha=0.01, epsilon=0.01):
        self.N = N
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.goal_state = (self.N - 1, self.N - 1)  # (19,19) for 20x20 grid
        self.start_state = (0, 0)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # Actions: up(-1,0), down(+1,0), left(0,-1), right(0,+1)
        self.ACTIONS = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Q-table: dictionary of dictionaries
        self.Q = {state: {action: 0 for action in self.ACTIONS.keys()} for state in self.states}

        # Policy (greedy after learning)
        self.policy = {state: random.choice(list(self.ACTIONS.keys())) for state in self.states}

    def get_actions(self):
        return list(self.ACTIONS.keys())

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.get_actions())
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def update_policy(self):
        for state in self.states:
            self.policy[state] = max(self.Q[state], key=self.Q[state].get)

    def plot_rewards(self, rewards):
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episodes vs Total Rewards (Q-Learning)")
        plt.grid(True)
        plt.savefig('qlearning_rewards.png')
        plt.show()

    def plot_steps(self, steps):
        plt.figure()
        plt.plot(steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps to Goal")
        plt.title("Episodes vs Steps to Goal (Q-Learning)")
        plt.grid(True)
        plt.savefig('qlearning_steps.png')
        plt.show()

    def plot_grid(self):
        value_grid = np.zeros((self.N, self.N))
        for state in self.states:
            value_grid[state] = max(self.Q[state].values())
        plt.figure()
        plt.imshow(value_grid, cmap='jet', origin='lower', vmin=-20, vmax=100)
        plt.colorbar(label='Max Q-value')
        plt.title('Learned Value Table')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('qlearning_value_table.png')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning for Maze Navigation")
    parser.add_argument('--grid_size', type=int, default=20, help='Size of the grid (NxN)')
    parser.add_argument('--alpha', type=restricted_float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=restricted_float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=restricted_float, default=0.01, help='Epsilon for exploration')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--loggin', action='store_true', help='Enable logging')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    args = parser.parse_args()

    # Initialize environment
    grid = Grid(args.grid_size, gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon)
    reward_history = []
    step_history = []

    for episode in range(args.episodes):
        state = grid.start_state
        total_reward = 0
        done = False

        for step in range(args.steps):
            # Choose action using epsilon-greedy policy
            action = grid.epsilon_greedy(state)

            # Compute next state
            dx, dy = grid.ACTIONS[action]
            next_state = (state[0] + dx, state[1] + dy)

            # Check boundaries
            if not (0 <= next_state[0] < grid.N and 0 <= next_state[1] < grid.N):
                reward = -10
                next_state = state  # Stay in same place
                done = True
            elif next_state == grid.goal_state:
                reward = 100
                done = True
            else:
                reward = -1
                done = False

            # Q-learning update
            best_next_q = max(grid.Q[next_state].values())
            td_target = reward + grid.gamma * best_next_q
            td_error = td_target - grid.Q[state][action]
            grid.Q[state][action] += grid.alpha * td_error

            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            # End episode if done
            if done:
                break

        reward_history.append(total_reward)
        step_history.append(step)

        if args.loggin and episode % 100 == 0:
            logging.info(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step}")

    # Final updates
    grid.update_policy()

    # Plot results
    if args.plot:
        grid.plot_rewards(reward_history)
        grid.plot_steps(step_history)
        grid.plot_grid()