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

# =============================================================================
# Dueling DQN
# Architecture that separates value and advantage estimation
# =============================================================================
class DuelingQNetwork(nn.Module):
    """
    Dueling DQN separates the value function into two streams:
    
    Q(s,a) = V(s) + A(s,a)
    
    - V(s): Value of being in state s
    - A(s,a): Advantage of taking action a
    
    This allows the network to learn which states are valuable
    without needing to learn the effect of each action.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingQNetwork, self).__init__()
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    
    def forward(self, x):
        features = self.features(x)
        # Get value and advantage
        value = self.value_stream(features) # V(s)
        advantage = self.advantage_stream(features) # A(s,a)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # Subtracting mean stabilizes training
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True)) # TODO
        
        return q_values
    

class DuelingDQNAgent:
    """
    Dueling DQN uses the dueling network architecture.
    The update rule is the same as DDQN, 
    but the network architecture is different.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        # TODO
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.001 
        self.epsilon_decay = 0.999 
        self.batch_size = 30
        self.target_update_freq = 500
        self.action_dim = action_dim

        # Networks (Dueling architecture)
        self.q_network = DuelingQNetwork(state_dim=state_dim, action_dim=action_dim).to(device) # TODO
        self.target_network = DuelingQNetwork(state_dim=state_dim, action_dim=action_dim).to(device) # TODO
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4) # TODO
        self.memory = ReplayBuffer(capacity=10000)
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
    
    def update(self):

        if len(self.memory) < self.batch_size:
            return None
        
        batch = self.memory.sample(self.batch_size) # TODO
        
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).to(device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # TODO
        
        # Target Q values
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q # TODO
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()