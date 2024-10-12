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
from torch.distributions import Normal

def initialize_agent(env, file_pth, model_args, optimizer_args, training_args, alg_args):
    if alg_args['type'] == 'DQN':
        return DQN_Agent(env, file_pth, model_args, optimizer_args, training_args, alg_args) 
    elif alg_args['type'] == 'VPG':
        return VPG_Agent(env, file_pth, model_args, optimizer_args, training_args, alg_args)
    else:
        raise NotImplementedError("Agent type is not implemented!")

class DQN_Agent():

    def __init__(self, env, file_pth, model_args, optimizer_args, training_args, alg_args):

        self.file_pth = file_pth

        self.model_args = model_args

        self.training_args = training_args

        self.alg_args = alg_args

        self.seed = training_args.get('seed', None) 

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            env.action_space.seed(self.seed)

        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = env.reset(seed=self.seed)

        state = preprocess_data(state, model_args)
        self.policy_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args, alg_args=alg_args).to(self.device)
        self.target_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args, alg_args=alg_args).to(self.device)
        
        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss

        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), **optimizer_args)
        
        self.replay_buffer = ReplayBufferDeque(capacity=training_args['buffer_size'], device=self.device, seed=self.seed) 

    def choose_action(self, env, state, epsilon):
        """
        Note: Pong has Box (continuous) action space, so actions can be tensors.
        Cartpole has Discrete action space, so actions can't be tensors.
        """
        if random.random() < epsilon:
            action = env.action_space.sample()
            # action = torch.tensor(action, dtype=torch.int64)
        else:
            action = torch.argmax(
                self.compute_qvals(state.unsqueeze(0)) # unsqueeze state to add batch axis since nn expects it
                ).squeeze(0).item() # revert batch axis and convert tensor to python scalar

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
                targets = rewards + discount * torch.max(
                    self.compute_qvals(next_states, True), dim=1
                )[0] * (1 - terminations) # Only return reward if not in a terminal state

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
    
    def train(self):

        writer = SummaryWriter(log_dir='runs/' + self.file_pth) # causes overwrite issues

        episodes = self.training_args['episodes']
        batch_size = self.training_args['batch_size']
        discount = self.training_args['discount']
        target_update_method = self.training_args['target_update_method']
        target_update_interval = self.training_args['target_update_interval']
        step_repeat = self.training_args['step_repeat']

        epsilon = self.training_args['epsilon']
        epsilon_min = self.training_args['epsilon_min']
        epsilon_min_ep = self.training_args['epsilon_min_ep']
        epsilon_decay = self.training_args['epsilon_decay']
        epsilon_start = epsilon
        epsilon_decay_exp = (epsilon_min / epsilon_start) ** (1 / epsilon_min_ep)
        
        step = 0
        best_reward = -9999999
        start_time = time.time()

        for episode in tqdm(range(episodes), desc="Training episodes"):
            state, _ = self.env.reset(seed=self.seed)

            state = preprocess_data(state, self.model_args)
            
            ep_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                step += 1

                action = self.choose_action(self.env, state.to(self.device), epsilon) 
                
                reward = 0

                for _ in range(step_repeat): 
                    
                    next_state, step_reward, terminated, truncated, _ = self.env.step(action)

                    reward += step_reward

                    if terminated:
                        break
                
                ep_reward += reward       
            
                next_state = preprocess_data(next_state, self.model_args)

                self.replay_buffer.add_transition(state, action, next_state, reward, terminated)
                
                loss = self.update_weights(discount, batch_size)
                
                state = next_state
                
                if step % target_update_interval == 0:

                    if target_update_method == 'soft':
                        self.soft_update_target(self.policy_nn, self.target_nn)
                    elif target_update_method == 'hard':
                        self.hard_update_target()
                    else:
                        raise NotImplementedError
            
            if not epsilon_decay:
                epsilon = max(epsilon_start * (epsilon_decay_exp ** episode), epsilon_min)
            else:
                epsilon = max(epsilon * epsilon_decay, epsilon_min)

            end_time = time.time()
            elapsed_time = end_time - start_time
            #print(f"\nCompleted episode {episode} with score {ep_reward} in {elapsed_time} seconds! Epsilon: {epsilon}.\n")

            if ep_reward > best_reward:
                best_reward = ep_reward
                print(f"\nBest episode so far! Completed episode {episode} with score {ep_reward}! Elapsed time: {elapsed_time} seconds. Epsilon: {epsilon}. Step: {step}.")

            self.policy_nn.save_model(self.file_pth)

            writer.add_scalar("reward", ep_reward, episode)
            if loss:
                writer.add_scalar("loss", loss, episode)
            writer.add_scalar('epsilon', epsilon, episode)
            
        self.env.close()
        writer.flush()
        writer.close()

    def test(self, model_path):
        self.policy_nn.load_model(model_path)

        state, _ = self.env.reset(seed=self.seed)

        state = preprocess_data(state, self.model_args)

        ep_reward = 0
        
        terminated = False

        truncated = False

        step = 0

        while not (terminated or truncated):
            
            action = self.choose_action(self.env, state.to(self.device), epsilon=0.05) 

            reward = 0

            step += 1
                
            for i in range(self.training_args['step_repeat']):

                next_state, step_reward, terminated, truncated, _ = self.env.step(action)

                reward += step_reward
    
                if terminated:
                    print(f"Game end at step: {step}, reward: {ep_reward}!")
                    break
            
            if step % 500 == 0:
                print(f"Step: {step}!")

            next_state = preprocess_data(next_state, self.model_args)
            state = next_state
            ep_reward += reward

class VPG_Agent():

    def __init__(self, env, file_pth, model_args, optimizer_args, training_args, alg_args):
        self.env = env
        self.file_pth = file_pth

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        self.model_args = model_args
        self.training_args = training_args
        self.alg_args = alg_args

        state, _ = self.env.reset()
        state = preprocess_data(state, model_args)
        
        if alg_args['continuous']:
            # multiply actions space dim by 2 for sd as well
            self.nn_policy = General_NN(state.shape, self.env.action_space.shape[0] * 2, model_args, alg_args).to(self.device)
        else:
            self.nn_policy = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)

        self.optimizer = initialize_optimizer(self.nn_policy.parameters(), **optimizer_args)
        
    def choose_action(self, state):
        """
        returns action and corresponding log_prob
        """
        # https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a
        # stochastic policy; sample from proportionally to the softmax (automatically applied) probabilities
        if self.alg_args['continuous']:
            action_params = self.nn_policy(state.unsqueeze(0))
            
            action_dim = self.env.action_space.shape[0]
            mu, std = action_params[:, :action_dim], torch.exp(action_params[:, action_dim:]) # sd cant be negative; this is state dependent implementation for sd

            dist = Normal(mu, std)
        else:
            dist = self.nn_policy(state)

        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def update_params(self, rewards, log_probs):
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #print(log_probs)
        loss = -torch.mean(log_probs * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_returns(self, rewards, discount):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = discount * G + reward
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
        return returns

    def train(self):
        
        writer = SummaryWriter(log_dir='runs/' + self.file_pth)

        episodes = self.training_args['episodes']
        discount = self.training_args['discount']

        for episode in tqdm(range(episodes), desc="Training episodes"):

            rewards = []
            log_probs = []
            state, _ = self.env.reset()
            
            terminated = False
            truncated = False

            ep_reward = 0

            while not (terminated or truncated):
                state = preprocess_data(state, self.model_args)

                action, log_prob = self.choose_action(state.to(self.device))
                
                next_state, reward, terminated, truncated, _ = self.env.step(np.array([action])) # try removing .item()
                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state

                ep_reward += reward
                #print(reward, "reward")
            
            returns = self.compute_returns(rewards, discount)
            self.update_params(returns, torch.stack(log_probs).to(self.device))

            self.nn_policy.save_model(self.file_pth)

            writer.add_scalar("reward", ep_reward, episode)

    def test(self, model_pth):
        
        self.nn_policy.load_model(model_pth)
        self.nn_policy.eval()
        #print(self.nn_policy.state_dict())

        state, _ = self.env.reset()

        state = preprocess_data(state, self.model_args)

        ep_reward = 0
        
        terminated = False

        truncated = False

        step = 0

        while not (terminated or truncated):
            
            action = self.choose_action(state.to(self.device)) 

            step += 1
                
            next_state, reward, terminated, truncated, _ = self.env.step(action)
    
            if terminated or truncated:
                print(f"Game end at step: {step}, reward: {ep_reward}!")

            next_state = preprocess_data(next_state, self.model_args)
            state = next_state
            ep_reward += reward

class DDPG_Agent():

    def __init__(self, env, file_pth, model_args, optimizer_args, training_args, alg_args):
        self.env = env

        self.file_pth = file_pth

        self.optimizer_args = optimizer_args

        self.training_args = training_args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = self.env.reset()

        self.buffer = ReplayBufferDeque(training_args['buffer_size'], self.device)

        self.q_nn1 = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.q_nn2 = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.q_nn1_targ = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.q_nn2_targ = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.q_nn1_targ.load_state_dict(self.q_nn1.state_dict())
        self.q_nn2_targ.load_state_dict(self.q_nn2.state_dict())

        self.policy_nn = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.policy_nn_targ = General_NN(state.shape, self.env.action_space.n, model_args, alg_args).to(self.device)
        self.policy_nn_targ.load_state_dict(self.policy_nn.state_dict())

        """
        self.q_optimizer = torch.optim.Adam(
            list(self.q_nn1.parameters()) + list(self.q_nn2.parameters()),
            lr=3e-4
        )
        """

    def choose_action(self):
        pass

    def update_q(self, states, actions, next_states, rewards, terminations):
        
        q_pred1 = self.q_nn1(states)[torch.arange(states.size(0)), actions.long()]
        q_pred2 = self.q_nn2(states)[torch.arange(states.size(0)), actions.long()]

        targ_actions = torch.clamp(
            self.policy_nn_targ(next_states) + torch.clamp(
                noise,
                min=-noise_clip,
                max=noise_clip
            ),
            min=-action_clip,
            max=action_clip
        )

        with torch.no_grad():
            # change gamma to alg_args
            q_targ = rewards + self.alg_args['gamma'] * min(
                self.q_nn1_targ(next_states, targ_actions),
                self.q_nn2_targ(next_states, targ_actions)
            ) * (1 - terminations)

        loss1 = self.loss_fn(q_pred1, q_targ)
        loss2 = self.loss_fn(q_pred2, q_targ)
        self.optimizer_q.zero_grad()
        loss1.backward()
        loss2.backward()
        self.optimizer_q.step()

    def update_policy(self, states):

        actions = self.policy_nn(states)
        loss = -torch.mean(self.q_nn1(states, actions))
        loss.backward()
        self.optimizer_policy.step()


    def train(self):
        
        episodes = self.training_args['episodes']
        batch_size = self.training_args['batch_size']
        
        for episode in range(episodes):
            
            state, _ = self.env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):

                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.buffer.add_transition(state, action, next_state, reward, terminated)

                if self.buffer.capacity >= batch_size:

                    for _ in range(number_of_updates):

                        states, actions, next_states, rewards, terminations = self.buffer.sample(batch_size)

                        loss_nn1, loss_nn2 = self.update_q(states, actions, next_states, rewards)

                        loss_policy = self.update_policy(states, actions)
                        if delaymet:







