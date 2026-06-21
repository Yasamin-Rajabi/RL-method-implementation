import numpy as np
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int = 100):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

# =============================================================================
# Prioritized Experience Replay (PER) DQN
# Based on "Prioritized Experience Replay" (Schaul et al., 2016)
# =============================================================================

class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    Stores priorities in a binary tree where:
    - Parent = sum of children
    - Leaf nodes = individual priorities
    
    Allows O(log n) sampling and O(1) priority updates.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity + 1) # Binary tree storage
        self.data = [None] * capacity # Actual transitions
        self.write_idx = 0 # Current write position
        self.size = 0 # Number of stored items
    
    def add(self, priority: float, transition: Transition):
        """Add new experience with given priority"""
        idx = self.capacity + self.write_idx 
        
        self.data[self.write_idx] = transition
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity) 
    
    def update(self, idx: int, priority: float):
        """Update priority of a leaf node"""
        self.tree[idx] = priority
        
        while idx > 0:
            idx = (idx - 1) // 2
            left = 2 * idx + 1
            right = 2 * idx + 2       
            self.tree[idx] = self.tree[left] + self.tree[right]
    
    def sample(self, batch_size: int) -> list:
        """Sample batch based on priorities (higher priority = more likely)"""
        batch = []
        segment = self.tree[0] / batch_size 
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            r = random.uniform(a, b) 
            idx = self._find_leaf(0, r)
            batch.append((self.data[idx - self.capacity], idx))
        
        return batch
    
    def _find_leaf(self, idx: int, r: float) -> int:
        """Find leaf node containing priority value r"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  
            return idx

        if r <= self.tree[left]:
            return self._find_leaf(left, r)
        else:
            return self._find_leaf(right, max(0, r - self.tree[left]))
    
    def __len__(self):
        return self.size
    
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Key ideas:
    - Not all experiences are equally important
    - Sample experiences with higher TD-error more often
    - Use importance sampling weights to correct bias
    
    Priority = |TD_error|^α + ε
    - α: how much prioritization (0 = uniform, 1 = full)
    - ε: small constant to ensure non-zero probability
    """
    
    def __init__(self, capacity: int = 100, alpha: float = None, beta: float = None):
        self.capacity = capacity
        self.alpha = alpha # Priority exponent
        self.beta = beta # Importance sampling exponent
        self.epsilon = 1e-5 # Small constant
        
        self.tree = SumTree(capacity) 
        self.max_priority = 1.0  
    
    def push(self, *args):
        """Add new experience with max priority (so it's sampled soon)"""
        transition = Transition(*args)
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> tuple:
        """Sample batch with priorities and compute IS weights"""
        sampled = self.tree.sample(batch_size)
        batch = [t for t, idx in sampled] 
        
        priorities = []
        indices = [idx for t, idx in sampled]
        
        for i, transition in enumerate(batch):
            leaf_idx = indices[i]
            priority = self.tree.tree[leaf_idx]   
            priorities.append(priority)
        
        priorities = np.array(priorities)
        probs = priorities / self.tree.tree[0]
        weights = (self.tree.size * probs) ** (-self.beta)
        weights = weights / weights.max()  
        
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])
    
    
        return states, actions, rewards, next_states, dones, weights, batch, indices
    
    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha # TODO
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.tree)