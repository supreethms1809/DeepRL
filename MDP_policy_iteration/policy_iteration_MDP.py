'''
Problem statement:
    - Gridworld of size NxN
    - Reward of -1 for each state
    - Reward of 100 for the goal state
    - Actions: up, down, left, right
    - Policy: Uniform random policy
    - Discount factor: 0.9
    - Actions are deterministic and actions leading out of the grid will stay in the same state

Usage:
    For V Function:
        python policy_iteration_MDP.py --grid_size 4 --use_V

    For Q Function:
        python policy_iteration_MDP.py --grid_size 4 

'''

import numpy as np
import argparse
import logging
import copy

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
    
    def initialize_valueFunction(self, grid):
        actions, ACTIONS = grid.get_actions()
        if self.use_V:
            valueFunc = {state: 0 for state in grid.states}
        else:
            valueFunc = {state: {action: 0 for action in actions} for state in grid.states}
        return valueFunc
    
class policy_V():
    def __init__(self, grid):
        self.grid = grid
        self.actions, self.ACTIONS = grid.get_actions()
        self.policy = {state: {action: 1/len(self.actions) for action in self.actions} for state in grid.states}
    
    def policy_evaluation(self, grid, value):
        V = copy.deepcopy(value)
        V_new = {}
        V_new = copy.deepcopy(V)
        convergence = False
        actions, ACTIONS = self.actions, self.ACTIONS
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
                
            #logging.info(f"-------------Iteration {k}-------------------")
            k += 1
            #logging.info(f"Value Function: {V_new}")
            # if k == 5:
            #     break
            if np.allclose(np.array(list(V.values())), np.array(list(V_new.values()))):
                convergence = True
                break
            V = copy.deepcopy(V_new)
        logging.info(f"Value converged in {k} iterations")
        
        # policy improvement        
        logging.info(f"*****************Starting Policy Improvement*************************")
        stable_policy =  copy.deepcopy(self.policy)
        for state in grid.states:
            vk_minus_1 = {a:0 for a in actions}
            for action in actions:
                if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= grid.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= grid.N):
                    next_state = state
                else:
                    next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                vk_minus_1[action] = V[next_state]
            #logging.info(f"State: {state}, V: {list(vk_minus_1.values())}")
            max_v = max(list(vk_minus_1.values()))
            max_act = [i for i, x in enumerate(list(vk_minus_1.values())) if x == max_v]
            actions = list(ACTIONS.keys())
            #logging.info(f"State: {state}, Actions: {[actions[i] for i in max_act]}")

            #logging.info(f"State: {state}, Policy: {self.policy[state]}")
            for action in actions:
               stable_policy[state][action] = 1/len(max_act) if action in [actions[i] for i in max_act] else 0
            #logging.info(f"State: {state}, Stable Policy: {stable_policy[state]}")

        return V, stable_policy

class policy_Q():
    def __init__(self, grid):
        self.grid = grid
        self.actions, self.ACTIONS = grid.get_actions()
        self.policy = {state: {action: 1/len(self.actions) for action in self.actions} for state in grid.states}
    
    def policy_evaluation(self, grid, value):
        Q_old = copy.deepcopy(value)
        Q_new = {}
        Q_new = copy.deepcopy(Q_old)
        convergence_threshold = 1e-10 
        convergence = False
        actions, ACTIONS = self.actions, self.ACTIONS
        k = 1
        #logging.info(f"Q Before iteration Function: {Q_old}")
        while True:
            #logging.info(f"-------------Iteration {k}-------------------")
            delta = 0
            # Policy Evaluation
            for state in grid.states:
                for action in actions:
                    if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= grid.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= grid.N):
                        next_state = state
                    else:
                        next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                    temp = 0
                    #logging.info(f"Before State: {state}, Action: {action}, Q_old: {Q_old[state][action]} ,Q_new: {Q_new[state][action]}")
                    for next_action in actions:
                        #logging.info(f"Before Inside Next Action: Q_old: {Q_old[next_state][next_action]}")
                        temp += self.policy[state][next_action] * Q_old[next_state][next_action]
                        #logging.info(f"After Inside Next Action: Q_old: {Q_old[next_state][next_action]}")
                    Q_new[state][action] = grid.rewards[state] + (grid.gamma * temp)
                    #logging.info(f"After update State: {state}, Action: {action}, Q_old: {Q_old[state][action]} ,Q_new: {Q_new[state][action]}")

                    # Calculate the maximum change for convergence check
                    delta = max(delta, abs(Q_new[state][action] - Q_old[state][action]))

            if delta < convergence_threshold:
                logging.info(f"Convergence achieved in {k} iterations")
                convergence = True
                break

            k += 1
            Q_old = copy.deepcopy(Q_new)
        
        # Policy Improvement
        logging.info(f"*****************Starting Policy Improvement*************************")
        stable_policy =  copy.deepcopy(self.policy)
        for state in grid.states:
            Qk_minus_1 = {a:0 for a in actions}
            for action in actions:
                if (state[0] + ACTIONS[action][0] < 0) or (state[0] + ACTIONS[action][0] >= grid.N) or (state[1] + ACTIONS[action][1] < 0) or (state[1] + ACTIONS[action][1] >= grid.N):
                    next_state = state
                else:
                    next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
                for next_action in actions:
                    Qk_minus_1[action] = Q_old[next_state][next_action]
            #logging.info(f"State: {state}, V: {list(Qk_minus_1.values())}")
            max_q = max(list(Qk_minus_1.values()))
            max_act = [i for i, x in enumerate(list(Qk_minus_1.values())) if x == max_q]
            actions = list(ACTIONS.keys())
            #logging.info(f"State: {state}, Actions: {[actions[i] for i in max_act]}")

            #logging.info(f"State: {state}, Policy: {self.policy[state]}")
            for action in actions:
               stable_policy[state][action] = 1/len(max_act) if action in [actions[i] for i in max_act] else 0
            #logging.info(f"State: {state}, Stable Policy: {stable_policy[state]}")

        return Q_old, stable_policy

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

    parser = argparse.ArgumentParser(description='Size of the grid, Create NxN grid')
    log = argparse.ArgumentParser(description='Logging level')
    parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid (NxN)')
    parser.add_argument('--use_V', action='store_true', help='Use V Function')
    args = parser.parse_args()
    N = args.grid_size
    use_V = args.use_V
    logging.info(f"user_V: {use_V}")

    if use_V:
        logging.info("Policy Iteration using V Function")
    else:
        logging.info("Policy Iteration using Q Function")

    # Initialize the grid
    grid = Grid(N)
    value = valueFunction(use_V=use_V)
    value = value.initialize_valueFunction(grid)

    if use_V:
        logging.info(f"*****************Policy Iteration using V*************************")
        policy = policy_V(grid)
        logging.info(f"[V] Current Policy: {policy.policy}")
        V, stable_policy = policy.policy_evaluation(grid, value)
        logging.info(f"[V] Stable Policy: {stable_policy}")
    else:
        logging.info(f"*****************Policy Iteration using Q*************************")
        policy = policy_Q(grid)
        logging.info(f"[Q] Current Policy Q: {policy.policy}")
        Q, stable_policy = policy.policy_evaluation(grid, value)
        # logging.info(f"[Q] Value Function: {Q}")
        logging.info(f"[Q] Stable Policy: {stable_policy}")

    # Test: Can we modify policy second time?
    if use_V:
        logging.info(f"***************** [2] Policy Iteration using V*************************")
        logging.info(f"[V] Current Policy: {stable_policy}")
        policy.policy = stable_policy
        V, stable_policy = policy.policy_evaluation(grid, value)
        logging.info(f"[V] New Stable Policy: {stable_policy}")
    else:
        logging.info(f"***************** [2] Policy Iteration using Q*************************")
        logging.info(f"[Q] Current Policy Q: {stable_policy}")
        policy.policy = stable_policy
        Q, stable_policy = policy.policy_evaluation(grid, value)
        # logging.info(f"[Q] Value Function: {Q}")
        logging.info(f"[Q] New Stable Policy: {stable_policy}")
    


