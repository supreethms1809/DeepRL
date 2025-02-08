'''
Problem statement:
    - Gridworld of size NxN
    - Reward of -1 for each state
    - Reward of 100 for the goal state
    - Actions: up, down, left, right
    - Policy: Uniform random policy
    - Discount factor: 0.9
    - Actions are deterministic and actions leading out of the grid will stay in the same state

'''


import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Todo: pass the grid_size to the environment
parser = argparse.ArgumentParser(description='Size of the grid, Create NxN grid')
log = argparse.ArgumentParser(description='Logging level')
parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid (NxN)')
parser.add_argument('--use_V', type=bool, default=True, help='Use V Function')
parser.add_argument('--use_Q', type=bool, default=False, help='Use Q Function')
args = parser.parse_args()
N = args.grid_size
use_V = args.use_V
use_Q = args.use_Q

if use_V:
    logging.info("Policy Iteration using V Function")
elif use_Q:
    logging.info("Policy Iteration using Q Function")
else:
    logging.error("Please select a valid function to use")

# Define the actions
ACTIONS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Define the grid
# Todo: Change this to get the grid size from the user
GRID_SIZE = N
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

if use_V:
    V_policy_state = {state: 0 for state in states}
    V_policy_state_new = {state: 0 for state in states}
    logging.info(f"V_policy_state: {V_policy_state}")
elif use_Q:
    Q_policy_state = {state: {action: 0 for action in actions} for state in states}
    logging.info(f"Q_policy_state: {Q_policy_state}")
else:
    logging.error("Please select a valid function to use")

eps = 1e-10

k = 1
while True:

# Policy Evaluation
# Calculate V function for the policy using Vk(s) = policy(a|s) * (reward + gamma * (prob(s'|s,a) * Vk-1(s')))
# policy (0, 0): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}

    delta = 0

    for s in states:
        current_state = s
        temp = 0
        for a in actions:
            # print (f"Current state: {s}, Action: {a}")
            # print (f"Policy: {policy[s][a]}")
            # print (f"Reward: {rewards[s]}")
            if (s[0] + ACTIONS[a][0] < 0) or (s[0] + ACTIONS[a][0] >= GRID_SIZE) or (s[1] + ACTIONS[a][1] < 0) or (s[1] + ACTIONS[a][1] >= GRID_SIZE):
                next_state = s
            else:
                next_state = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
            # print (f"Next State: {next_state}")
            temp += policy[s][a] * (rewards[s] + gamma * V_policy_state[next_state])
            # print (f"Temp: {temp}")
        V_policy_state_new[s] = temp

    if np.array_equal(V_policy_state, V_policy_state_new):
        logging.info(f"Policy Evaluation converged in {k} iterations")
        logging.info(f"V_policy_state: {V_policy_state}")
        break
    else: 
    # Copy V_new to V for next iteration
        V_policy_state = V_policy_state_new.copy()
        k += 1

    # Policy Improvement
    improved_policy = {state: {action: 1/len(actions) for action in actions} for state in states}

optimal_policy = {state: None for state in states}  

logging.info(f"Optimal Policy: {optimal_policy}")

