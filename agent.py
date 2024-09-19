import numpy as np
from model import DQN_cnn, DQN_dnn, General_NN
import torch 
import torch.nn as nn
from replaybuffer import ReplayBufferDeque, ReplayBufferManual
import random
from model import initialize_optimizer, initialize_loss
from utils import preprocess_data, preprocess_data2
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

class DQN_Agent():
    def __init__(self, env, model_args, optimizer_args):

        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = env.reset()
        state = preprocess_data(self.device, state)
        self.policy_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)
        self.target_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)

        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss

        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), **optimizer_args) #torch.optim.Adam(self.policy_nn.parameters(), lr=1e-3) # TRY CHANGING TO ADAM FOR FASTER
        
        self.replay_buffer = ReplayBufferManual(capacity=500000, state_shape=state.shape, device=self.device)

    def choose_action(self, env, state, epsilon):
        if random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64)
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

    def hard_update_target(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def soft_update_target(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_weights(self, discount, batch_size):
        loss = None
        if self.replay_buffer.counter >= batch_size:
            states, actions, next_states, rewards, terminations = self.replay_buffer.sample(batch_size)
            
            #states = torch.stack(states) # states = (64, 1, 210, 160), (batchsize, in_channels, h, w)
            #next_states = torch.stack(next_states)
            #rewards = torch.stack(rewards)
            #actions = torch.stack(actions) #- 2 # current actions are 2 and 3 so change to 0 and 1 for indexing
            #terminations = torch.stack(terminations)
            
            preds = self.compute_qvals(states)[torch.arange(states.size(0)), actions.long()] # convert actions to long since they're used as indices
            
            with torch.no_grad():
                targets = rewards + discount * torch.max(self.compute_qvals(next_states, True), dim=1)[0] * (1 - terminations) # we only want to return reward if terminated state
            
            loss = self.loss_fn(preds, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def train(self, training_args):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=f'runs/pong_{timestamp}') #CHANGE

        batch_size = training_args['batch_size']
        discount = training_args['discount']
        target_update_interval = training_args['target_update_interval']
        epsilon = training_args["epsilon"]
        epsilon_min = training_args["epsilon_min"]
        #epsilon_min_ep = training_args["epsilon_min_ep"]
        epsilon_decay = training_args["epsilon_decay"]#(epsilon_min / epsilon_start) ** (1 / self.epsilon_min_ep)#0.9995

        step = 0
        best_reward = -9999999

        for episode in tqdm(range(training_args['episodes']), desc="Training episodes"):
            state, _ = self.env.reset()
        
            state = preprocess_data(self.device, state)
            
            ep_reward = 0

            start_time = time.time()

            terminated = False

            while not terminated:
                step += 1
                
                action = self.choose_action(self.env, state.to(self.device), epsilon) 

                reward = 0

                for i in range(4):

                    next_state, step_reward, terminated, truncated, _ = self.env.step(action)

                    reward += step_reward

                    if terminated:
                        break
                
                ep_reward += reward

                #reward = torch.tensor(reward, dtype=torch.float, device=self.device)
                #terminated = torch.tensor(terminated, dtype=torch.float, device=self.device)           
            
                next_state = preprocess_data(self.device, next_state)

                self.replay_buffer.add_transition(state, next_state, reward, action, terminated)
                
                loss = self.update_weights(discount, batch_size)
                
                state = next_state

                if step % target_update_interval == 0:
                    self.soft_update_target(self.target_nn, self.policy_nn)
            
            #if episode % 1 == 0 and episode > batch_size: # change so that it averages over that 100 interval
            writer.add_scalar("reward", ep_reward, episode)
            writer.add_scalar("loss", loss, episode)
            writer.add_scalar('epsilon', epsilon, episode)

            if ep_reward > best_reward:
                best_reward = ep_reward
                print(f"Best episode so far! Episode: {episode}, Reward: {ep_reward}, Epsilon: {epsilon}")
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Completed episode {episode} with score {ep_reward} in {elapsed_time} seconds!")

        self.env.close()
        writer.flush()
        writer.close()
           
class DQN_Agent2():
    def __init__(self, env, model_args, optimizer_args):

        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = env.reset()
        state = preprocess_data2(self.device, state)
        self.policy_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)
        self.target_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)

        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss

        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), **optimizer_args) #torch.optim.Adam(self.policy_nn.parameters(), lr=1e-3) # TRY CHANGING TO ADAM FOR FASTER
        
        self.replay_buffer = ReplayBufferDeque(capacity=500000)

    def choose_action(self, env, state, epsilon):
        if random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=self.device)
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

    def hard_update_target(self):
        self.target_nn.load_state_dict(self.policy_nn.state_dict())

    def soft_update_target(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_weights(self, discount, batch_size):
        loss = None
        if self.replay_buffer.counter >= batch_size:
            states, next_states, rewards, actions, terminations = zip(*self.replay_buffer.sample(batch_size))

            states = torch.stack(states) # (1, 1, 210, 160)
            next_states = torch.stack(next_states)
            rewards = torch.stack(rewards)
            actions = torch.stack(actions) #- 2 # current actions are 2 and 3 so change to 0 and 1 for indexing
            terminations = torch.stack(terminations)
            
            preds = self.compute_qvals(states)[torch.arange(states.size(0)), actions]

            with torch.no_grad():
                targets = rewards + discount * torch.max(self.compute_qvals(next_states, True), dim=1)[0] * (1 - terminations) # we only want to return reward if terminated state
            
            loss = self.loss_fn(preds, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss
    
    def train(self, training_args):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir=f'runs/pong_{timestamp}') #CHANGE

        batch_size = training_args['batch_size']
        discount = training_args['discount']
        target_update_interval = training_args['target_update_interval']
        
        epsilon = training_args["epsilon"]
        epsilon_min = training_args["epsilon_min"]
        #epsilon_min_ep = training_args["epsilon_min_ep"]
        epsilon_decay = training_args["epsilon_decay"]#(epsilon_min / epsilon_start) ** (1 / self.epsilon_min_ep)#0.9995

        start_time = time.time()
        step = 0
        best_reward = -9999999

        for episode in tqdm(range(training_args['episodes']), desc="Training episodes"):
            state, _ = self.env.reset()
        
            state = preprocess_data2(self.device, state)
            
            ep_reward = 0
            while True:
                step += 1

                action = self.choose_action(self.env, state, epsilon) 
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                ep_reward += reward

                reward = torch.tensor(reward, dtype=torch.float, device=self.device)
                terminated = torch.tensor(terminated, dtype=torch.float, device=self.device)           
            
                next_state = preprocess_data2(self.device, next_state)

                self.replay_buffer.add_transition(state, next_state, reward, action, terminated)
                
                loss = self.update_weights(discount, batch_size)
                
                state = next_state

                if step % target_update_interval == 0:
                    self.hard_update_target()

                if terminated or truncated:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"step: {step}")
                    break
            
            if episode % 1 == 0 and episode > batch_size: # change so that it averages over that 100 interval
                writer.add_scalar("reward", ep_reward, episode)
                writer.add_scalar("loss", loss, episode)

            if episode % 1000 == 0:
                print(episode, best_reward)
                
            if ep_reward > best_reward:
                best_reward = ep_reward
                print("episode", episode, "reward", ep_reward, "epsilon", epsilon, "best", best_reward)
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            writer.add_scalar("Epsilon vs. Episodes", epsilon, episode)

        self.env.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        torch.save({
            'model_state_dict': self.policy_nn.state_dict(),
            'optimizer_state_dict': self.policy_nn.state_dict(),
        }, "checkpoint.pth")
        writer.flush()
        writer.close()