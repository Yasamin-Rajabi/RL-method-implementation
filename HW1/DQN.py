import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    """Simple neural network to approximate Q(s,a)"""
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),   
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim) 
        )
    
    def forward(self, x):
        return self.network(x)
    
class DQNAgent:
    """
    Simple DQN Agent
    
    Key components:
    - Q-Network: approximates Q(s,a)
    - Target Network: provides stable targets for learning
    - Replay Buffer: stores past experiences
    - Epsilon-greedy: balances exploration/exploitation
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        # Hyperparameters
        self.gamma = 0.99 #TODO  Discount factor
        self.epsilon = 1.0 # TODO # Exploration rate
        self.epsilon_min = 0.001 # TODO # Minimum exploration
        self.epsilon_decay = 0.999 # TODO
        self.batch_size = 30 # TODO
        self.target_update_freq = 500 # TODO
        self.action_dim = action_dim
        
        # Networks
        self.q_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device) # TODO
        self.target_network = QNetwork(state_dim=state_dim, action_dim=action_dim).to(device) # TODO
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and memory
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4) # TODO
        self.memory = ReplayBuffer(capacity=10000) # TODO
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        # TODO
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    
    def store_transition(self, state, action, reward, next_state, done):
        # TODO
        self.memory.push(state, action, reward, next_state, done)
        pass
    
    def update(self):
        """Update Q-network using DQN algorithm"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        batch = self.memory.sample(self.batch_size) # TODO
        
        # TODO
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).to(device)

        
        # ================================================================
        # DQN Update (Standard)
        # Q_target = r + γ * max_a Q_target(s', a)
        # ================================================================
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # TODO
        
        # Target Q values
        with torch.no_grad():
            # Standard DQN: take max over all actions
            next_q = self.target_network(next_states).max(1)[0] # TODO
            target_q = rewards + (1 - dones) * self.gamma * next_q # TODO
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # TODO

        return loss.item()