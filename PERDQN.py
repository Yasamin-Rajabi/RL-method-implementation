import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import gymnasium as gym
# MP4 Libraries
from IPython.display import HTML
from base64 import b64encode
import cv2
import numpy as np

class PERDQNAgent:
    """
    DQN with Prioritized Experience Replay
    
    Differences from DDQN:
    - Uses SumTree for efficient prioritized sampling
    - Computes importance sampling weights
    - Updates priorities after each batch
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.gamma = 0.99 # TODO
        self.epsilon = 1.0 # TODO
        self.epsilon_min = 0.01 # TODO
        self.epsilon_decay = 0.9999 # TODO
        self.batch_size = 30 # TODO
        self.target_update_freq = 250 # TODO
        self.action_dim = action_dim
        
        # PER parameters
        self.alpha = 0.6 # TODO
        self.beta = 0.4 # TODO
        self.beta_increment = 1e-5 # TODO
        
        # Networks
        self.q_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)  # TODO
        self.target_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device)  # TODO
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4) # TODO
        
        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=8192, alpha=self.alpha, beta=self.beta)
        self.steps = 0
    
    def select_action(self, state, training=True):
        # TODO
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update_beta(self):
        """Anneal beta towards 1.0 and sync with buffer"""
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.memory.beta = self.beta  # keep buffer in sync

    def update(self):
        """PER DQN update with importance sampling"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from prioritized buffer
        states, actions, rewards, next_states, dones, weights, batch, indices = self.memory.sample(self.batch_size)# TODO
        
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # TODO
        
        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True) # TODO
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1) # TODO
            target_q = rewards + (1 - dones) * self.gamma * next_q # TODO
        
        # Compute TD errors for priority update
        td_errors = (current_q.detach() - target_q).abs().cpu().numpy()
        
        # Weighted loss (importance sampling)
        #weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device) # TODO
        loss = (weights * (current_q - target_q).pow(2)).mean() # TODO
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory.update_priorities(indices, td_errors)
        # Update priorities (using random indices for simplicity)
        # In full implementation, you'd track actual indices
        # self.memory.update_priorities(indices, td_errors)
        
        # Increment beta (towards 1.0)
        self.update_beta() # TODO
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()