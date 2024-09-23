import numpy as np
from model import DQN_cnn, DQN_dnn, General_NN
import torch 
import torch.nn as nn
from replaybuffer import ReplayBufferDeque, ReplayBufferManual, ReplayBufferDeque2
import random
from model import initialize_optimizer, initialize_loss
from utils import preprocess_data
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import yaml
import os
import cv2

class DQN_Agent():
    def __init__(self, env, model_args, optimizer_args):
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = env.reset()
        state = preprocess_data(state)
        self.policy_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)
        self.target_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)

        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss

        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), **optimizer_args)
        
        self.replay_buffer = ReplayBufferDeque(capacity=500000, device=self.device) 

    def choose_action(self, env, state, epsilon):
        if random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64)
        else:
            # unsqueeze state to add batch axis since nn expects it. revert after.
            action = torch.argmax(self.compute_qvals(state.unsqueeze(0))).squeeze(0)
        return action
        
    def compute_qvals(self, state, target=False):
        if target:
            q_values = self.target_nn(state)
        else:
            q_values = self.policy_nn(state)
        return q_values

    def update_weights(self, discount, batch_size):
        loss = None
        if self.replay_buffer.counter >= batch_size:
            states, actions, next_states, rewards, terminations = self.replay_buffer.sample(batch_size)
            
            preds = self.compute_qvals(states)[torch.arange(states.size(0)), actions.long()]

            with torch.no_grad():
                targets = rewards + discount * torch.max(self.compute_qvals(next_states, True), dim=1)[0] * (1 - terminations) # we only want to return reward if terminated state
            
            loss = self.loss_fn(preds, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def hard_update_target(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def soft_update_target(self, source_nn, target_nn, tau=0.005):
        for target_param, param in zip(target_nn.parameters(), source_nn.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def train(self, training_args, model_name):

        writer = SummaryWriter(log_dir='runs/' + model_name)
    
        batch_size = training_args['batch_size']
        discount = training_args['discount']
        target_update_interval = training_args['target_update_interval']

        epsilon = training_args["epsilon"]
        epsilon_min = training_args["epsilon_min"]
        epsilon_min_ep = training_args["epsilon_min_ep"]
        #epsilon_decay = training_args["epsilon_decay"]#(epsilon_min / epsilon_start) ** (1 / self.epsilon_min_ep)#0.9995
        epsilon_start = epsilon
        epsilon_decay = (epsilon_min / epsilon_start) ** (1 / epsilon_min_ep)
        
        step = 0
        best_reward = -9999999
        start_time = time.time()

        for episode in tqdm(range(training_args['episodes']), desc="Training episodes"):
            state, _ = self.env.reset()
            state = preprocess_data(state)
            
            ep_reward = 0
            terminated = False
            
            while not terminated:
                step += 1

                action = self.choose_action(self.env, state.to(self.device), epsilon) 
                
                reward = 0

                for _ in range(4):

                    next_state, step_reward, terminated, truncated, _ = self.env.step(action)

                    reward += step_reward

                    if terminated:
                        break
                
                ep_reward += reward       
            
                next_state = preprocess_data(next_state)

                self.replay_buffer.add_transition(state, action, next_state, reward, terminated)
                
                loss = self.update_weights(discount, batch_size)
                
                state = next_state

                if step % target_update_interval == 0:
                    self.soft_update_target(self.policy_nn, self.target_nn)
            
            #epsilon = max(epsilon * epsilon_decay, epsilon_min)
            epsilon = max(epsilon_start * (epsilon_decay ** episode), epsilon_min)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nCompleted episode {episode} with score {ep_reward} in {elapsed_time} seconds! Epsilon: {epsilon}.\n")

            if ep_reward > best_reward:
                best_reward = ep_reward
                print(f"\nBest episode so far!", end="")

            self.policy_nn.save_model(model_name)

            writer.add_scalar("reward", ep_reward, episode)
            writer.add_scalar("loss", loss, episode)
            writer.add_scalar('epsilon', epsilon, episode)
            
        self.env.close()
        writer.flush()
        writer.close()

    def test(self, model_path):
        self.policy_nn.load_model(model_path)

        state, _ = self.env.reset()
        state = preprocess_data(state)

        ep_reward = 0
        
        terminated = False

        while not terminated:
            
            action = self.choose_action(self.env, state.to(self.device), epsilon=0.05) 

            reward = 0
                
            for i in range(4):

                next_state, step_reward, terminated, truncated, _ = self.env.step(action)

                reward += step_reward

                frame = self.env.env.env.render() # unwrap wrappers

                resized_frame = cv2.resize(frame, (500, 400))

                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("Pong AI", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)

                if terminated:
                    break

            state = preprocess_data(next_state)

            ep_reward += reward