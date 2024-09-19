from collections import deque
import random
import numpy as np
import torch

class ReplayBufferDeque():
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.counter = 0

    def add_transition(self, state, next_state, reward, action, terminated):
        self.buffer.append((state, next_state, reward, action, terminated))
        self.counter += 1

    def sample(self, batch_size):
        """
        output: [(state1, next_state1, reward1, action1, terminated1), 
        (state2, next_state2, reward2, action2, terminated2), ...]
        """
        return random.sample(self.buffer, batch_size)

class ReplayBufferDeque2():
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.counter = 0

        self.device = device

    def add_transition(self, state, next_state, reward, action, terminated):
        self.buffer.append((state, next_state, reward, action, terminated))
        self.counter += 1

    def sample(self, batch_size):
        """
        output: [(state1, next_state1, reward1, action1, terminated1), 
        (state2, next_state2, reward2, action2, terminated2), ...]
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, next_states, rewards, actions, terminations = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        terminations = torch.tensor(np.array(terminations), dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, terminations
    
class ReplayBufferManual():
    def __init__(self, capacity, state_shape, device='cpu'):
        self.capacity = capacity
        self.counter = 0

        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8) #dtype=np.float32
        self.actions = np.zeros(self.capacity, dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminations = np.zeros(self.capacity, dtype=np.float32)

        self.device = device

    def add_transition(self, state, next_state, reward, action, terminated):
        idx = self.counter % self.capacity
        
        self.states[idx] = state
        self.actions[idx] = torch.tensor(action).detach().cpu() #action
        self.next_states[idx] = next_state
        self.rewards[idx] = reward
        self.terminations[idx] = terminated
        
        self.counter = min(self.counter + 1, self.capacity)

    def sample(self, batch_size):
        """
        output: [(state1, next_state1, reward1, action1, terminated1), 
        (state2, next_state2, reward2, action2, terminated2), ...]
        """
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
