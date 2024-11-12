from collections import deque
import random
import numpy as np
import torch

class ReplayBufferDeque():
    """
    Complete deque-based replay buffer.
    """

    def __init__(self, capacity, device='cpu', seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.size = 0

        self.device = device
        random.seed(seed)

    def add_transition(self, state, action, next_state, reward, terminated):
        self.buffer.append((state, action, next_state, reward, terminated))
        self.size += 1

    def sample(self, batch_size):
        """
        Note: 'state', 'next_state' are already tensors
        but the rest are not, so make sure to convert them to tensors. 
        Also, 'states' and 'next_states' are a tuple of tensors, so use 
        torch.stack to convert it to a single tensor. 
        Edit: actions may be tuple of tensors so torch.stack may be preferrable. 
        DO NOT use torch.tensor for them as you'll get the following error:
        
        "ValueError: only one element tensors can be converted to Python scalars"
        """
        batch = random.choices(self.buffer, k=batch_size)# .choices uses w/ replace; use .sample for without repl
        
        states, actions, next_states, rewards, terminations = zip(*batch) # e.g. states = (state1, state2, state3, ...)
        
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).float().to(self.device)#torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(self.device)
        
        return states, actions, next_states, rewards, terminations

class ReplayBufferDeque2():
    """
    A simple deque-based replay buffer. [Obsolete]
    
    Note: This implementation requires separate tensor conversion 
    and cuda transfer.
    """
    
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.counter = 0

    def add_transition(self, state, action, next_state, reward, terminated):
        self.buffer.append((state, next_state, reward, action, terminated))
        self.counter += 1

    def sample(self, batch_size):
        """
        output: [(state1, action1, next_state1, reward1, terminated1), 
        (state2, action2, next_state2, reward2, terminated2), ...]
        """
        return random.sample(self.buffer, batch_size)
    
class ReplayBufferManual():
    """
    A replay buffer that uses numpy arrays, pre-initialized with the
    size as the underlying data structure.
    """

    def __init__(self, capacity, state_shape, device='cpu'):
        self.capacity = capacity
        self.counter = 0

        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8) #dtype=np.float32
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminations = np.zeros(self.capacity, dtype=np.float32)

        self.device = device

    def add_transition(self, state, action, next_state, reward, terminated):
        idx = self.counter % self.capacity
        
        self.states[idx] = state
        self.actions[idx] = torch.tensor(action).detach().cpu() 
        self.next_states[idx] = next_state
        self.rewards[idx] = reward
        self.terminations[idx] = terminated
        
        self.counter = min(self.counter + 1, self.capacity)

    def sample(self, batch_size):
        batch = np.random.choice(self.counter, batch_size) # run w/ replacement (default)

        states = self.states[batch]
        actions = self.actions[batch]
        next_states = self.next_states[batch]
        rewards = self.rewards[batch]
        terminations = self.terminations[batch]
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, terminations
