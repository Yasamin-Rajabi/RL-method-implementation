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

class DDQNAgent(DQNAgent):
    """
    Double DQN extends DQN to reduce overestimation.
    
    The problem with standard DQN:
    - Uses max() which tends to overestimate Q-values
    
    Double DQN solution:
    - Use online network to SELECT the action
    - Use target network to EVALUATE the action
    
    This decouples selection from evaluation.
    """
    
    def update(self):
        """Update using Double DQN algorithm"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        batch = self.memory.sample(self.batch_size) # TODO
        
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).to(device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)# TODO
        
        with torch.no_grad():
            # Step 1: Use ONLINE network to select best action
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True) # TODO
            
            # Step 2: Use TARGET network to evaluate that action
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1) # TODO
            
            # Step 3: Compute target
            target_q =  rewards + (1 - dones) * self.gamma * next_q # TODO
        
        # Compute loss and update (same as DQN)
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()