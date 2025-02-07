import numpy as np
import argparse
import logging

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the actions
ACTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Define the grid
# Todo: Change this to get the grid size from the user
GRID_SIZE = 4
logging.info(f"Initializing Gridworld with size {GRID_SIZE}x{GRID_SIZE}")
GRID = np.zeros((GRID_SIZE, GRID_SIZE))

# Define the states in the grid
states = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]

# Define the rewards
rewards = {state: -1 for state in states}
rewards[(GRID_SIZE-1, GRID_SIZE-1)] = 100  # Reward for Goal
gamma = 0.9
logging.info(f"Reward: {rewards}")

# Define the policy
actions = list(ACTIONS.keys())
policy = {state: {action: 1/len(actions) for action in actions} for state in states}
logging.info(f"Policy: {policy}")

# Main function
if __name__ == "__main__":

    # # Todo: pass the grid_size to the environment
    # parser = argparse.ArgumentParser(description='Size of the grid, Create NxN grid')
    # log = argparse.ArgumentParser(description='Logging level')
    # parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid (NxN)')
    # parser.add_argument('--log', type=str, default=None, help='Logging level')
    # args = parser.parse_args()
    # N = args.grid_size
    # print("Grid Size: ", N)

    logging.info("Policy Iteration using Value Function")