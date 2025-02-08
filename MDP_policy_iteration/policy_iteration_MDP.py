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

class Grid():
    def __init__(self, N, gamma=0.9):
        self.N = N
        self.GRID = np.zeros((self.N, self.N))
        self.states = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.rewards = {state: -1 for state in self.states}
        self.rewards[(self.N-1, self.N-1)] = 100  # Reward for Goal
        self.gamma = gamma

    def get_actions(self):
        ACTIONS = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        return list(ACTIONS.keys()), ACTIONS

class valueFunction():
    def __init__(self, use_V=True):
        self.use_V = use_V
    
    def initialize_valueFunction(self, grid,actions):
        if self.use_V:
            valueFunc = {state: 0 for state in grid.states}
        else:
            valueFunc = {state: {action: 0 for action in actions} for state in grid.states}
        return valueFunc
    
class policy_V():
    def __init__(self, grid, actions):
        self.policy = {state: {action: 1/len(actions) for action in actions} for state in grid.states}
    
    def policy_evaluation(self, grid, value, actions):
        V = value.copy()
        V_new = value.copy()
        convergence = False
        
        k = 1
        while True:
            # Policy Evaluation
            for state in grid.states:
                temp = 0
                for action in actions:
                    if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= grid.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= grid.N):
                        next_state = state
                    else:
                        next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                    temp += self.policy[state][action] * (grid.rewards[state] + grid.gamma * V_new[next_state])
                
                V_new[state] = temp
                
            logging.info(f"-------------Iteration {k}-------------------")
            logging.info(f"Value Function: {V_new}")
            k += 1
            if k == 5:
                break
            # if np.allclose(np.array(list(V.values())), np.array(list(V_new.values()))):
            #     convergence = True
            #     break
            V = V_new.copy()
        
        # policy
        logging.info(f"Value converged in {k} iterations")
        stable_policy =  self.policy.copy()
        for state in grid.states:
            vk_minus_1 = {a:0 for a in actions}
            for action in actions:
                if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= grid.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= grid.N):
                    next_state = state
                else:
                    next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                vk_minus_1[action] = V[next_state]
            logging.info(f"State: {state}, V: {list(vk_minus_1.values())}")
            max_v = max(list(vk_minus_1.values()))
            max_act = [i for i, x in enumerate(list(vk_minus_1.values())) if x == max_v]
            logging.info(f"State: {state}, Actions: {max_act}")

            #stable_policy[state][] = 1

        return V, stable_policy

    def policy_improvement(self, grid, value, stable_policy):
        return stable_policy

class policy_Q():
    def __init__(self, grid, actions):
        self.policy = {state: {action: 1/len(actions) for action in actions} for state in grid.states}
    
    def policy_evaluation(self, grid, value):
        Q = value.copy()
        Q_new = value.copy()
        convergence = False
        
        k = 1
        while True:

            # Policy Evaluation
            for state in grid.states:
                for action in actions:
                    Q_new[state][action] = grid.rewards[state] + grid.gamma * sum([Q[state][action] for state in grid.states])

            k += 1
            Q = Q_new.copy()
            if np.allclose(Q, Q_new):
                convergence = True
                break
        
        # Policy

        return Q, self.policy

    def policy_improvement(self, grid, value):
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

    parser = argparse.ArgumentParser(description='Size of the grid, Create NxN grid')
    log = argparse.ArgumentParser(description='Logging level')
    parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid (NxN)')
    parser.add_argument('--use_V', type=bool, default=True, help='Use V Function')
    args = parser.parse_args()
    N = args.grid_size
    use_V = args.use_V

    if use_V:
        logging.info("Policy Iteration using V Function")
    else:
        logging.info("Policy Iteration using Q Function")

    grid = Grid(N)
    actions, ACTIONS = grid.get_actions()
    value = valueFunction()
    value = value.initialize_valueFunction(grid, ACTIONS)

    if use_V:
        policy = policy_V(grid, actions)
        V, stable_policy = policy.policy_evaluation(grid, value, actions)
        #logging.info(f"Value Function: {V}")
        updated_policy = policy.policy_improvement(grid, value, stable_policy)
        optimal_policy = policy.policy
    else:
        policy = policy_Q(grid, actions)
        Q = policy.policy_evaluation(grid, value)
        logging.info(f"Value Function: {Q}")
        policy.policy_improvement(grid, value)
        optimal_policy = policy.policy
    


