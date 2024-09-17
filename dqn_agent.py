import numpy as np
from nn_model import DQN_cnn, DQN_dnn, GeneralDQN
import torch 
import torch.nn as nn
from replaybuffer import ReplayBufferDeque
import random
from utils import initialize_optimizer, initialize_loss

class DQN_Agent():
    def __init__(self, device, action_dim, state_dim, model_args, optimizer_args, training_args):
        self.device = device
        
        self.policy_nn = GeneralDQN(state_dim, action_dim, model_args).to(device)
        self.target_nn = GeneralDQN(state_dim, action_dim, model_args).to(device)
        
        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss
        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), **optimizer_args) #torch.optim.Adam(self.policy_nn.parameters(), lr=1e-3) # TRY CHANGING TO ADAM FOR FASTER
        self.replay_buffer = ReplayBufferDeque(capacity=100000)

        self.batch_size = training_args["batch_size"]
        self.discount = training_args["discount"]
        
        self.epsilon_start = 1.0
        self.epsilon_min = 0.01
        self.epsilon_min_ep = training_args["epsilon_min_ep"]
        self.epsilon_decay = (self.epsilon_min / self.epsilon_start) ** (1 / self.epsilon_min_ep)#0.9995
        self.epsilon = self.epsilon_start

        self.actions = range(action_dim)
        self.target_update_interval = 100#10000

    def choose_action(self, env, state, device=None):
        if random.random() < self.epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            # unsqueeze state to add batch axis since nn expects it. revert after.
            action = torch.argmax(self.compute_qvals(state.unsqueeze(0))).squeeze(0)

        #self.epsilon = max(self.epsilon - 1.1e-6, 0.01)
        return action
        
    def compute_qvals(self, state, target=False):
        if target:
            q_values = self.target_nn(state)
        else:
            q_values = self.policy_nn(state)
        return q_values

    def update_target(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def update_weights(self):
        loss = None
        if self.replay_buffer.counter >= self.batch_size:
            states, next_states, rewards, actions, terminations = zip(*self.replay_buffer.sample(self.batch_size))

            states = torch.stack(states) # (1, 1, 210, 160)
            next_states = torch.stack(next_states)
            rewards = torch.stack(rewards)
            actions = torch.stack(actions) #- 2 # current actions are 2 and 3 so change to 0 and 1 for indexing
            terminations = torch.stack(terminations)
            
            preds = self.compute_qvals(states)[torch.arange(states.size(0)), actions]

            with torch.no_grad():
                targets = rewards + self.discount * torch.max(self.compute_qvals(next_states, True), dim=1)[0] * (1 - terminations) # we only want to return reward if terminated state
            
            loss = self.loss_fn(preds, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

        
           