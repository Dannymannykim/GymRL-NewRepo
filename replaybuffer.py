from collections import deque
import random
import numpy as np

class ReplayBufferDeque():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.counter = 0

    def add_state(self, state):
        self.buffer.append(state)
        self.counter += 1

    def sample(self, batch_size):
        """
        output: [(state1, next_state1, reward1, action1, terminated1), 
        (state2, next_state2, reward2, action2, terminated2), ...]
        """
        return random.sample(self.buffer, batch_size)

class ReplayBufferManual():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.counter = 0
        self.idx = 0 

        self.states = np.zeros((capacity, 210, 160), dtype=np.float32)
        self.next_states = np.zeros((capacity, 210, 160), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.terminations = np.zeros(capacity, dtype=np.bool_)

    def add_state(self, state, next_state, reward, action, terminated):
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.actions[self.idx] = action
        self.terminations[self.idx] = terminated
        self.counter = min(self.counter + 1, self.capacity)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        """
        output: [(state1, next_state1, reward1, action1, terminated1), 
        (state2, next_state2, reward2, action2, terminated2), ...]
        """
        indices = np.random.choice(self.counter, batch_size, replace=False)
        return self.states[indices], self.next_states[indices], self.rewards[indices], self.actions[indices], self.terminations[indices]