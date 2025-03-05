from calendar import c
from email.policy import default
import logging
from math import log
import re
from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader 
import gymnasium as gym
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_size, output_size, use_dense=True):
        super().__init__()
        self.use_dense = use_dense
        if use_dense:
            self.denseQ = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            )
        else:
            self.convQ = nn.Sequential(
                nn.Conv2d(input_size, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 96 * 96, 512),
                nn.ReLU(),
                nn.Linear(512, output_size)
            )

        with torch.no_grad():
            if use_dense:
                self.denseQ_target = self.denseQ
            else:
                self.convQ_target = self.convQ

        self.optimizer = AdamW(self.parameters(), lr=0.000001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        if self.use_dense:
            x = self.denseQ(x)
        else:
            x = self.convQ(x)
        return x
    
    def updateQTarget(self):
        with torch.no_grad():
            if self.use_dense:
                Qtarget = self.denseQ_target = self.denseQ
            else:
                Qtarget = self.convQ_target = self.convQ
        return Qtarget

'''
# # I might not need this
# class RQN(nn.Module):
#     super().__init__()
#     def __init__(self, input_size, output_size):
#         self.densePolicy = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )
#         self.optimizer = AdamW(self.parameters(), lr=0.001)
#         self.loss_fn = nn.MSELoss()

#     def forward(self, x):
#         x = self.densePolicy(x)
#         return x
'''
    
class ReplayBuffer(nn.Module):
    def __init__(self, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, next_state, reward):
        bufferFull = True
        tup = (state, action, next_state, reward)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(tup)
            self.position += 1
            bufferFull = False
        return bufferFull

    # TODO: Use yield function to return a batch of data
    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def check_buffer(self):
        logger.info(f"Buffer size: {len(self.buffer)}")
        if len(self.buffer) == self.buffer_size:
            return True
        else:
            return False
        
    def flush(self):
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        logger.info(f"Length of buffer: {self.buffer}")
        return len(self.buffer)

class DQNAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.obs = env.reset()[0]
        self.action_space = env.action_space
        self.state_space = env.observation_space
        # self.Qnw = DQN(self.state_space.shape[0], self.action_space.n)
        # self.Qnw_target = self.Qnw.updateQTarget()
        # self.replay_buffer = ReplayBuffer(1000)

    def get_action(self, current_obs, Qnw):
        if np.random.uniform(0,1) < 0.01:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                x = Qnw(current_obs)
                x = x[0]
            return torch.argmax(x,dim=0).item()

    def computeTarget(self):
        pass

    def updateQweights(self):
        pass

if __name__ == "__main__":
    # Starting RL training 
    logger.info("Starting RL training")
    batch_size = 64
    env_name = "CarRacing-v3"
    env = gym.make(env_name, lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode=None)
    #env = gym.make(env_name, render_mode="human")

    obs, info = env.reset()
    logger.info(f"Observation space: {env.observation_space}, Action space: {env.action_space}")

    # Initialize the agent
    agent = DQNAgent(env).to(device)
    Qnw = DQN(env.observation_space.shape[2], env.action_space.n, use_dense=False)
    
    # Initialize the replay buffer
    buffer_size = 1024
    replay_buffer = ReplayBuffer(buffer_size).to(device)
    
    num_episodes = 1000
    num_steps = 100000
    M_update = 512
    reward_plot_values = []
    steps_plot_values = []
    loss_plot_values = []

    Qnw.to(device)

    for episode in tqdm(range(num_episodes)):
        m = 0
        total_reward = 0
        total_loss = 0
        total_steps = 0
        done = False
        current_obs, info = env.reset()
        current_obs = torch.tensor(current_obs, dtype=torch.float32)

        current_obs = torch.stack([current_obs for _ in range(batch_size)], dim=0)
        current_obs = current_obs.reshape(current_obs.shape[0], current_obs.shape[3], current_obs.shape[2], current_obs.shape[1])
        current_obs = current_obs.to(device)
        while not done:
            bufferFull = False
            for step in range(num_steps):
                #logger.info(f"Episode: {episode}, Step: {step}")

                action = agent.get_action(current_obs, Qnw)
                next_obs, reward, terminated, truncated, info = agent.env.step(action)
                current_obs = torch.unbind(current_obs, dim=0)
                current_obs = current_obs[0]
                bufferFull = replay_buffer.add(current_obs, action, next_obs, reward)
                m += 1
                
                if m == M_update:
                    m = 0
                    #logger.info("Updating Q target in {} steps".format(M_update))
                    Q_target = Qnw.updateQTarget()
                    Q_target = Q_target.to(device)
                next_obs = [torch.tensor(next_obs, dtype=torch.float32) for _ in range(batch_size)]
                next_obs = torch.stack(next_obs)
                next_obs = next_obs.reshape(next_obs.shape[0], next_obs.shape[3], next_obs.shape[2], next_obs.shape[1])
                next_obs = next_obs.to(device)
                current_obs = next_obs
                total_steps += 1

                if bufferFull:
                    #logger.info("Buffer is full!!! Starting training")
                    # Draw a batch from the replay buffer
                    # Do batch processing
                    #batch_size = 32
                    num_batches = buffer_size // batch_size
                    batch = [None] * num_batches
                    for i in range(num_batches):
                        batch_loss = 0
                        batch_reward = 0
                        batch[i] = replay_buffer.get_batch(batch_size)

                        # Perform a batch update
                        current_obs, action, next_obs, reward = zip(*batch[i])

                        current_obs = torch.stack(current_obs)
                        current_obs = current_obs.to(device)
                        next_obs = [torch.tensor(obs, dtype=torch.float32) for obs in next_obs]
                        next_obs = torch.stack(next_obs)
                        next_obs = next_obs.to(device)
                        next_obs = next_obs.reshape(next_obs.shape[0], next_obs.shape[3], next_obs.shape[2], next_obs.shape[1])
                        action = torch.tensor(action, dtype=torch.int64).to(device)
                        reward = torch.tensor(reward, dtype=torch.float32).to(device)

                        batch_Q_target = Q_target(next_obs)
                        current_Q_values = Qnw(current_obs)
                        max_Q_target_value = batch_Q_target.max(dim=1)[0]
                        target = reward + 0.95 * max_Q_target_value
                        action_mask = torch.nn.functional.one_hot(action, num_classes=Qnw(current_obs).shape[1])
                        predicted_Q = (current_Q_values * action_mask).sum(dim=1)
                        loss = Qnw.loss_fn(predicted_Q, target)
                        Qnw.optimizer.zero_grad()
                        loss.backward()
                        Qnw.optimizer.step()
                        batch_reward += reward.sum().item()
                        batch_loss += loss.item()
                    
                    total_reward += float(batch_reward)
                    total_loss += float(batch_loss)
                    #logger.info("Buffer training done !!!!!Flushing the buffer")
                    replay_buffer.flush()
                    
                    ## Perform individual updates for each item in the batch
                    #     for current_obs, action, next_obs, reward in batch:
                    #         # Convert next observation to tensor
                    #         next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
                    #         next_obs = next_obs.reshape(1, next_obs.shape[0], next_obs.shape[1], next_obs.shape[2])
                    #         Q_target = Q_target.to(device)
                    #         Q_target_value = Q_target(next_obs)
                    #         max_Q_target_value = Q_target_value.max(dim=1)[0].item()  # Get the maximum Q-value along the action dimension
                    #         target = reward + 0.95 * max_Q_target_value

                    #         # Calculate input and Output for loss
                    #         currentQ_value = Qnw(current_obs)
                    #         targetQ_value = Q_target_value.clone().detach()
                    #         targetQ_value[0][action] = target

                    #         loss = Qnw.loss_fn(currentQ_value, targetQ_value)
                    #         Qnw.optimizer.zero_grad()
                    #         loss.backward()
                    #         Qnw.optimizer.step()
                
                    #         batch_reward += reward
                    #         batch_loss += loss.item()
                    #     total_steps += 1
                    #     total_reward += batch_reward
                    #     total_loss += batch_loss
                    # replay_buffer.flush()

                if terminated or truncated:
                    done = True
                    break

            if terminated or truncated:
                done = True
                break

        if episode % 1 == 0:
            logger.info(f"Episode: {episode}, Total reward: {total_reward} Total loss: {total_loss}")
            reward_plot_values.append(total_reward)
            loss_plot_values.append(total_loss)
            steps_plot_values.append(total_steps)
    logger.info("Training complete")
    # Save model
    torch.save(Qnw.state_dict(), "Qnw"+str(datetime.datetime.now())+".pth")
    logger.info("Model saved")
    logger.info("Plotting the results")
    env.close()
    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(reward_plot_values)
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.title("Episodes vs Total reward")
    plt.savefig("reward_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    plt.plot(loss_plot_values)
    plt.xlabel("Episodes")
    plt.ylabel("Total loss")
    plt.title("Episodes vs Total loss")
    plt.savefig("loss_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    plt.plot(steps_plot_values)
    plt.xlabel("Episodes")
    plt.ylabel("Total steps")
    plt.title("Episodes vs Total steps")
    plt.savefig("steps_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    # # Visualize the env
    # env = gym.make(env_name, lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode="human")
    # obs, info = env.reset()
    # # load the model
    # Qnw.load_state_dict(torch.load("Qnw.pth"))
    # Qnw.eval()
    # Qnw.to(device)
    # logger.info("Starting visualization")
    # # Initialize the agent
    # agent = DQNAgent(env).to(device)
    
    # env.close()
    # logger.info("Visualization complete")