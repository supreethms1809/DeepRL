import numpy as np
import random

# Define the maze environment
class MazeEnv:
    def __init__(self, size):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        self.actions = ['up', 'down', 'left', 'right']
    
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)
        
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self.state, reward, done

    def get_possible_actions(self):
        return self.actions

# Define the epsilon-greedy policy
def epsilon_greedy_policy(V, state, epsilon, env):
    if np.random.rand() < epsilon:
        return random.choice(env.get_possible_actions())
    else:
        best_action = None
        best_value = -np.inf
        for action in env.get_possible_actions():
            next_state, _, _ = simulate_step(env, state, action)
            if V[next_state] > best_value:
                best_value = V[next_state]
                best_action = action
        return best_action

# Simulate a step without changing the environment state
def simulate_step(env, state, action):
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(env.size - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(env.size - 1, y + 1)
    next_state = (x, y)
    reward = 1 if next_state == env.goal else -0.1
    done = next_state == env.goal
    return next_state, reward, done

# TD(0) learning algorithm
def td_learning(env, num_episodes, alpha, gamma, epsilon):
    V = { (x, y): 0 for x in range(env.size) for y in range(env.size) }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy_policy(V, state, epsilon, env)
            next_state, reward, done = env.step(action)
            
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    
    return V

# Extract the policy from the value function
def extract_policy(V, env):
    policy = {}
    for x in range(env.size):
        for y in range(env.size):
            state = (x, y)
            best_action = None
            best_value = -np.inf
            for action in env.get_possible_actions():
                next_state, _, _ = simulate_step(env, state, action)
                if V[next_state] > best_value:
                    best_value = V[next_state]
                    best_action = action
            policy[state] = best_action
    return policy

# Find the shortest path using the extracted policy
def find_shortest_path(policy, env):
    state = env.reset()
    path = [state]
    while state != env.goal:
        action = policy[state]
        state, _, _ = simulate_step(env, state, action)
        path.append(state)
    return path

# Parameters
size = 20
num_episodes = 10000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# Create the environment
env = MazeEnv(size)

# Perform TD learning
V = td_learning(env, num_episodes, alpha, gamma, epsilon)

# Extract the policy
policy = extract_policy(V, env)

# Find the shortest path
shortest_path = find_shortest_path(policy, env)
print("Shortest path:", shortest_path)