import os
from venv import logger
from networkx import selfloop_edges
import numpy as np
import argparse
import logging
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = 'cuda'
    logging.info("CUDA is available. Running on Nvidia GPU")
elif torch.mps.is_available():
    device = 'mps'
    logging.info("MPS is available. Running on Apple M4 GPU")
else:
    device = 'cpu'
    logging.info("CUDA and MPS are not available. Running on CPU")

# Racing environment
class RacingEnv(nn.Module):
    def __init__(self, env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=100):
        nn.Module.__init__(self)
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 48 * 48, 512)).to(device)
        
        self.critic = nn.Sequential(
            self.features,
            nn.Linear(512, output_dim)).to(device)
        
        self.actor = nn.Sequential(
            self.features,
            nn.Linear(512, output_dim)).to(device)

        self.Aoptimizer = optim.AdamW(self.parameters(), lr=self.alpha, weight_decay=1e-2, betas=(0.9, 0.99))
        #self.Coptimizer = optim.AdamW(self.parameters(), lr=self.alpha, weight_decay=1e-2, betas=(0.9, 0.99))
        self.Aloss = nn.MSELoss()
        self.targetQ = self.actor.copy()

    def forward(self, x):
        """
        Default forward method returning both actor and critic outputs.
        """
        actor_out = self.actor(x)
        critic_out = self.critic(x)
        return actor_out, critic_out

    # Q network
    def forward_actor(self, x):
        """
        Forward pass for the actor network.
        """
        return self.actor(x)

    # Policy network
    def forward_critic(self, x):
        """
        Forward pass for the critic network.
        """
        return self.critic(x)
    
    def updatetargetQ(self, source_network):
        """
        Copy the weights from source_network to targetQ.
        """
        self.targetQ.load_state_dict(source_network.state_dict())
        return self.targetQ
    
    def get_Q_target_state_action(self, state, action):
        y = self.targetQ(state).gather(1, action.unsqueeze(1))
        return y
    
    def get_Q_state_action(self, state, action):
        y = self.Q(state).gather(1, action.unsqueeze(1))
        return y
    
# Define Agent
class RacingAgent(RacingEnv):
    def __init__(self, env, input_dim, output_dim, epsilon=0.01, gamma=0.9, alpha=0.01, episodes=100, batch_size=32):
        RacingEnv.__init__(self, env, input_dim, output_dim, epsilon, gamma, alpha, episodes)
        self.batch_size = batch_size
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.episodes = episodes
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            t_action = torch.tensor([self.env.action_space.sample() for _ in range(state.shape[0])], device=device)
        else:
            #logger.info(f"Policy: {self.policy(state).argmax(dim=1)}")
            with torch.no_grad():
                t_action = self.critic(state).argmax(dim=1)
        return t_action

    def train(self):
        loss_per_episode = []
        steps_per_episode = []
        reward_per_episode = []
        ############### Training loop ################
        for episode in range(self.episodes):
            tot_loss = 0
            tot_reward = 0
            tot_step = 0
            batch_size = self.batch_size
            state = self.env.reset()
            state = torch.tensor(state[0], dtype=torch.float32).to(device)
            state = torch.permute(state, (2, 1, 0))
            done = False
            step = 0

            logger.info(f"Episode: {episode}")
            # Train the Q network
            while not done:
                action = self.get_action(state)
                action_array = action
                action_array = action_array.cpu().numpy()
                for n in range(self.batch_size):
                    ns[n], re[n], terminated_array, truncated_array, _ = self.env.step(action_array[n])
                if np.any(terminated_array) or np.any(truncated_array):
                    done = True
                ns_t = torch.tensor(ns, dtype=torch.float32).to(device)
                ns_t = torch.permute(ns_t, (0, 3, 1, 2)).contiguous()
                re_t = torch.tensor(re, dtype=torch.float32).to(device).contiguous()
                #logger.info(f"Reward: {re_t.shape}")

                next_action = self.get_action(ns_t)

                with torch.no_grad():
                    next_q = self.get_Q_target_state_action(ns_t, next_action)
                    Q_target = re_t + self.gamma * next_q
                    # logger.info(f"Q_target: {Q_target.shape}")

                # Calculate TD_error
                q_values = self.get_Q_state_action(state, action).contiguous()
                TD_error = q_values - Q_target
                Qloss = self.Qloss(q_values, Q_target)

                self.Qoptimizer.zero_grad()
                Qloss.backward()
                self.Qoptimizer.step()

                # Update policy network
                action_probs = self.policy(state)
                selected_action_probs = action_probs.gather(1, action.unsqueeze(1))
                policy_loss = -(torch.log(selected_action_probs) * q_values.detach()).mean()
                self.policyoptimizer.zero_grad()
                policy_loss.backward()
                self.policyoptimizer.step()

                # Update state
                state = ns_t
                step += 1

                # Update target Q network
                if step % 10 == 0:  
                    self.updatetargetQ(self.Q)
                tot_loss += Qloss.item()
                tot_reward += re_t.sum()
            
            loss_per_episode.append(tot_loss)
            steps_per_episode.append(step)
            reward_per_episode.append(tot_reward)
            if episode % 10 == 0:
                logger.info(f"Episode: {episode}, Loss: {tot_loss}, Steps: {step}, Reward: {tot_reward}")

        return loss_per_episode, steps_per_episode, reward_per_episode

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == "__main__":

    epsilon = 0.01
    gamma = 0.9
    alpha = 0.01
    episodes = 2000
    batch_size = 32
    run = True

    env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode=None)
    input_dim = env.observation_space.shape[2]
    output_dim = env.action_space.n
    logger.info(f"Input channels: {input_dim}, Output dimension: {output_dim}")
    agent = RacingAgent(env, input_dim, output_dim, epsilon, gamma, alpha, episodes, batch_size)
    agent.to(device)
    loss_per_episode, steps_per_episode, reward_per_episode = agent.train()
    agent.save("model.pth")
    
    # Plot the results
    import matplotlib.pyplot as plt
    import datetime
    plt.plot(reward_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.title("Episodes vs Total reward")
    plt.savefig("reward_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    plt.plot(loss_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total loss")
    plt.title("Episodes vs Total loss")
    plt.savefig("loss_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    plt.plot(steps_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total steps")
    plt.title("Episodes vs Total steps")
    plt.savefig("steps_plot"+str(datetime.datetime.now())+".png")
    plt.show()

    env.close()

    logger.info("Training completed")
    
    if run:
        env = gym.make("CarRacing-v3", lap_complete_percent=0.95, domain_randomize=False, continuous=False, render_mode="human")
        state = env.reset()
        state = torch.tensor(state[0], dtype=torch.float32).to(device)
        state = torch.permute(state, (2, 1, 0))
        state = state.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        done = False
        while not done:
            action = agent.get_action(state)
            action_array = action
            action_array = action_array.cpu().numpy()
            ns, re, terminated_array, truncated_array, _ = env.step(action_array[0])
            if np.any(terminated_array) or np.any(truncated_array):
                done = True
            ns_t = torch.tensor(ns, dtype=torch.float32).to(device)
            ns_t = torch.permute(ns_t, (0, 3, 1, 2)).contiguous()
            state = ns_t
        env.close()
        logger.info("Testing completed")
