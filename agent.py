import numpy as np
from model import DQN_cnn, DQN_dnn, General_NN, Critic
import torch 
import torch.nn as nn
from buffer import ReplayBufferDeque, ReplayBufferManual, ReplayBufferDeque2, ReplayBufferSB3Style
import random
from model import initialize_optimizer, initialize_loss
from utils import preprocess_data
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from torch.distributions import Normal
from ou_noise import OU_Noise
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import vector

def initialize_agent(env, type, file_pth, model_args, parameters):
    """
    Initializes a RL agent based on given settings.

    Args:
        file_pth (str): Full path to save configs and models.
        model, optimizer, training, algorithm args

    Returns:
        Agent Class: The initialized agent instance.
    """
    agent_classes = {
        'DQN': DQN_Agent,
        'VPG': VPG_Agent,
        'TD3': TD3_Agent
    }
    
    Agent = agent_classes.get(type)

    if Agent is None:
        raise NotImplementedError("Agent type is not implemented!")

    return Agent(env, file_pth, model_args, **parameters)

class DQN_Agent():

    """
    Args:
        env (gym.Env): The environment to train on.
        file_pth (str): Directory path to save models, logs, and configurations.
        model_args (dict): Arguments for constructing the actor and critic networks.
        lr (float): Learning rate for the q-network. Defaults to 1e-3.
        buffer_size (int): Maximum size of the replay buffer. Defaults to 1,000,000.
        learning_starts (int): Number of steps before training starts and transitions are sampled. Defaults to 100.
        batch_size (int): Size of minibatch sampled from the replay buffer. Defaults to 256.
        tau (float): Target smoothing coefficient (Polyak averaging). Defaults to 0.005.
        gamma (float): Discount factor for future rewards. Defaults to 0.99.
        optimizer_type (str): Optimizer to use ('Adam', 'SGD', etc.). Defaults to 'Adam'.
        epsilon (float): Initial epsilon for epsilon-greedy exploration. Defaults to 1.0.
        epsilon_min (float): Minimum epsilon value. Defaults to 0.01.
        epsilon_min_ep (int): Episode at which epsilon reaches epsilon_min. Defaults to None.
        epsilon_decay (float): Multiplicative decay factor for epsilon per episode. Defaults to 0.99.
        step_repeat (int): Number of times to repeat each action. Defaults to 1.
        target_update_interval (int): Steps between target network updates. Defaults to 1.
        target_update_method (str): Update method ('hard' or 'soft'). Defaults to 'hard'.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    def __init__(
            self, 
            env, 
            file_pth, 
            model_args,
            lr=0.001,
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=256,
            tau=1,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            optimizer_type='Adam',
            epsilon=1,
            epsilon_min=0.01,
            exp_frac=0.1,
            step_repeat=1,
            target_update_interval=1,
            grad_norm_max=10,
            seed=None
    ):
        self.env = env
        self.file_pth = file_pth
        self.model_args = model_args
        self.lr = lr
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.optimizer_type = optimizer_type
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.exp_frac = exp_frac
        self.step_repeat = step_repeat
        self.target_update_interval = target_update_interval
        self.grad_norm_max = grad_norm_max
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            env.action_space.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = env.reset()
        state = preprocess_data(state, model_args)
        self.policy_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)
        self.target_nn = General_NN(state_dim=state.shape, action_dim=env.action_space.n, model_args=model_args).to(self.device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        #self.target_nn.eval()  # Optional: avoid updating BatchNorm/dropout during target usage
        
        self.loss_fn = initialize_loss(model_args["loss"]) #nn.MSELoss() # consider Huber loss
        self.optimizer = initialize_optimizer(self.policy_nn.parameters(), self.optimizer_type, self.lr)
        
        self.buffer = ReplayBufferDeque(capacity=self.buffer_size, device=self.device, seed=self.seed) 

        self.n_updates = 0 # for debugging

    def choose_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            env: The environment, which provides the action space.
            state (torch.Tensor): The current state, shape (state_dim,) or (c, h, w).
            epsilon (float): The probability of choosing a random action.

        Returns:
            Action (torch.Tensor): The chosen discrete action.

        Note: DQNs only work with discrete action spaces.
        Gym envs take in actions as int so make sure to change them later.
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            action = torch.tensor(action)
        else:
            # unsqueeze state to add batch axis since Pytorch nn expects it (CNN-type states will raise mismatch errors otherwise)
            with torch.no_grad():
                q_values = self.compute_qvals(state.unsqueeze(0)) 
                # revert batch axis and convert tensor to python scalar [outdated]; change description
                action = torch.argmax(q_values).squeeze(0).detach().cpu()
            
        return action
        
    def compute_qvals(self, state, target=False):
        if target:
            q_values = self.target_nn(state)
        else:
            q_values = self.policy_nn(state)

        return q_values
    
    def update_q(self, states, actions, next_states, rewards, terminations):
        """
        Performs a TD3-style update of the Q-networks using target policy smoothing.

        Args:
            states (torch.Tensor): Batch of current states, shape (batch_size, state_dim).
            actions (torch.Tensor): Batch of actions taken, shape (batch_size,).
            next_states (torch.Tensor): Batch of next states, shape (batch_size, state_dim).
            rewards (torch.Tensor): Batch of rewards received, shape (batch_size,).
            terminations (torch.Tensor): Batch of done flags (1 if terminal), shape (batch_size,).

        Returns:
            losses (torch.Tensor): The loss values for critic1 and critic2 respectively.

        
        """
        # States is of shape torch.Size([batch, features]) but actions is of shape torch.Size([batch]). Convert actions to torch.Size([batch, 1])
    
        preds = self.compute_qvals(states)[torch.arange(states.size(0)), actions.long()]
        
        with torch.no_grad():
            next_q_vals= torch.max(self.compute_qvals(next_states, True), dim=1)[0]
            targets = rewards + next_q_vals * (1 - terminations) * self.gamma

        loss = self.loss_fn(preds, targets) # Huber loss (less sensitive to outliers)
        """
        if self.n_updates == 5000:
            
            print(f'\n buffer_states: {states.sum(), next_states.sum(), actions.sum()}\n'
                f' current_q_vals: {preds.sum()}\n'
                f' target q_vals: {targets.sum()}\n'
                f' loss: {loss}\n' 
                f' next q_vals: {next_q_vals.sum()}\n'
                f' policy weights: {self.target_nn.fc_layers[0].weight.sum(), self.policy_nn.fc_layers[0].weight.sum()}\n'
                f' buffer_pos: {self.buffer.size}\n'
                f' rewards: {rewards.sum()}\n'
                f' dones: {terminations.sum()}\n'
                f' random_state: {np.random.get_state()[2]}')
            #print(states.sum(), preds[:6], targets[:6], loss, next_q_vals[:6], self.policy_nn.fc_layers[4].weight.sum(), self.buffer.size, 'ds')
            raise ImportError
            """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 10)
        self.optimizer.step()
        
        return loss

    def soft_update_target(self, source_nn, target_nn):
        with torch.no_grad():
            for target_param, param in zip(target_nn.parameters(), source_nn.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, total_steps, num_envs=None):

        writer = SummaryWriter(log_dir='runs/' + self.file_pth)
        
        episode = 0
        best_rew_mean = -999999
        start_time = time.time()

        # Call with seed only the first time to avoid recurring episodes.
        state, _ = self.env.reset(seed=self.seed) 
        state = preprocess_data(state, self.model_args)
        ep_reward = 0
        ep_rewards = []
        done = False

        for step in tqdm(range(total_steps), desc="Training steps"):
            action = self.choose_action(state.to(self.device), self.epsilon, step) 
                
            reward = 0
            # May be pointless with vectorized env
            for _ in range(self.step_repeat): 
                next_state, step_reward, terminated, truncated, _ = self.env.step(action.numpy())
                reward += step_reward
                done = terminated or truncated

                if done:
                    break
            
            ep_reward += reward     
        
            next_state = preprocess_data(next_state, self.model_args)

            # Flag done as True only for terminated, so truncated doesn't affect target value.
            self.buffer.add_transition(state, action, next_state, reward, done and not truncated) 
            
            # Soft (or Polyak) update. Set tau = 1 for hard updates.
            if (step >= self.learning_starts) and (step % self.target_update_interval == 0):
                self.soft_update_target(self.policy_nn, self.target_nn)
                
            if (step >= self.learning_starts) and (step % self.train_freq == 0): # removed batch size requirement to align with SB3 implementation
                losses = []
                for _ in range(self.gradient_steps):
                    # Buffer size can be < batch size as we sample w/ replacement.
                    states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)
                    loss = self.update_q(states, actions, next_states, rewards, dones)
                    losses.append(loss.item())
                    self.n_updates += 1

                writer.add_scalar("loss", np.mean(losses), step)
                
            if step >= self.learning_starts:
                # Update epsilon only after learning starts.
                self.epsilon = max(self.epsilon_min, 
                                1.0 - (1.0 - self.epsilon_min) * 
                                min(1.0, step / (self.exp_frac * total_steps)))
                
            state = next_state

            if done:
                end_time = time.time()
                elapsed_time = end_time - start_time

                ep_rewards.append(ep_reward)  
                if len(ep_rewards) > 100:
                    ep_rewards.pop(0)
                ep_rew_mean = sum(ep_rewards) / len(ep_rewards)
                if int(ep_rew_mean) > best_rew_mean:
                    best_rew_mean = int(ep_rew_mean)
                    print(f"\nBest reward so far! Completed episode {episode} with score {best_rew_mean}! "
                          f"Elapsed time: {elapsed_time} seconds. Epsilon: {self.epsilon}. Step: {step}.")
                    self.policy_nn.save_model(self.file_pth)

                writer.add_scalar("reward_mean", ep_rew_mean, step)
                writer.add_scalar('epsilon', self.epsilon, step)

                episode += 1
                
                state, _ = self.env.reset()
                state = preprocess_data(state, self.model_args)
                ep_reward = 0
                done = False
 
        self.env.close()
        writer.flush()
        writer.close()

    def test(self, model_path):
        self.policy_nn.load_model(model_path)

        state, _ = self.env.reset()
        state = preprocess_data(state, self.model_args)

        ep_reward = 0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            action = self.choose_action(state.to(self.device), epsilon=0.05) 

            reward = 0
            step += 1
                
            for i in range(self.step_repeat):
                next_state, step_reward, terminated, truncated, _ = self.env.step(action.numpy())
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
    """
    A Vanilla Policy Gradient agent.
     
    Note: The agent works on both discrete and continuous action spaces.
    """

    def __init__(
            self, 
            env, 
            file_pth, 
            model_args, 
            lr=0.001,
            gamma=0.99,
            optimizer_type='Adam',
            seed=None
    ):
        self.env = env
        self.file_pth = file_pth
        self.model_args = model_args
        self.lr = lr
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        state, _ = self.env.reset()
        state = preprocess_data(state, model_args)
        
        if model_args['continuous']:
            # multiply actions space dim by 2 for sd as well
            self.nn_policy = General_NN(state.shape, self.env.action_space.shape[0] * 2, model_args).to(self.device)
        else:
            model_args['output_transform'] = 'categorical'
            self.nn_policy = General_NN(state.shape, self.env.action_space.n, model_args).to(self.device) #Note: returns dist NOT tensor

        self.optimizer = initialize_optimizer(self.nn_policy.parameters(), self.optimizer_type, self.lr)
        
    def choose_action(self, state):
        """
        returns action and corresponding log_prob
        """
        # https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a
        # stochastic policy; sample from proportionally to the softmax (automatically applied) probabilities
        if self.model_args['continuous']:
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

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = self.gamma * G + reward
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
        return returns

    def train(self, episodes):
        
        writer = SummaryWriter(log_dir='runs/' + self.file_pth)

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
            
            returns = self.compute_returns(rewards)
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

class TD3_Agent():
    """
    A Twin-Delayed DDPG agent.
    
    Args:
        env (gym.Env): The environment to train on.
        file_pth (str): Directory path to save models, logs, and configurations.
        model_args (dict): Arguments for constructing the actor and critic networks.

        lr_critic (float): Learning rate for the critic network. Defaults to 1e-3.
        lr_actor (float): Learning rate for the actor network. Defaults to 1e-3.
        buffer_size (int): Maximum size of the replay buffer. Defaults to 1,000,000.
        learning_starts (int): Number of steps before training starts and transitions are sampled. Defaults to 100.
        batch_size (int): Size of minibatch sampled from the replay buffer. Defaults to 256.
        tau (float): Target smoothing coefficient (Polyak averaging). Defaults to 0.005.
        gamma (float): Discount factor for future rewards. Defaults to 0.99.
        optimizer_type (str): Optimizer to use ('Adam', 'SGD', etc.). Defaults to 'Adam'.
        train_freq (int): Number of steps between gradient updates. Defaults to 1.
        gradient_steps (int): Number of gradient updates per training step. Defaults to 1.
        noise_type (str): Type of exploration noise to add to actions ('Gaussian' or 'Ornstein-Uhlenbeck'). Defaults to 'Gaussian'.
        noise_std (float): Standard deviation of the exploration noise. Defaults to 0.1.
        policy_delay (int): Frequency of delayed policy updates relative to critic updates. Defaults to 2.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Note: This agent is intended to work with only continuous action spaces.
    """

    def __init__(
            self, 
            env, 
            file_pth, 
            model_args, 
            lr_critic=1e-3,
            lr_actor=1e-3,
            buffer_size=1_000_000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            optimizer_type='Adam',
            train_freq=1,
            gradient_steps=1,
            noise_type='Gaussian',
            noise_std=0.1,
            policy_delay=2,
            seed=None,
    ):
        self.env = env
        self.file_pth = file_pth
        self.model_args = model_args
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.policy_delay = policy_delay
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            env.action_space.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if isinstance(self.env, vector.VectorEnv): # check if vectorized env (specifically stable baseline's VecEnv class)
            state, _ = self.env.reset()
            state = preprocess_data(state, model_args)
            state_dim = state.shape[1:] # consider just using env.observation_space
            action_dim = self.env.action_space.shape[1] # may need to change if action space is multiple dimension as in shape (n, m),  (n, m, o), etc.
            action_max = float(self.env.action_space.high[0][0])
        else:
            state, _ = self.env.reset(seed=self.seed)
            state = preprocess_data(state, model_args)
            state_dim = state.shape
            action_dim = self.env.action_space.shape[0]
            action_max = float(self.env.action_space.high[0])

        #state = preprocess_data(state, model_args)
         
        if not model_args['continuous']:
            raise ValueError("The TD3 agent only works on continuous action spaces. Consider DQN or VPG for discrete.")

        self.critic1 = Critic(state_dim, action_dim, model_args).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, model_args).to(self.device)
        self.critic1_targ = Critic(state_dim, action_dim, model_args).to(self.device)
        self.critic2_targ = Critic(state_dim, action_dim, model_args).to(self.device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict()) # use hard update method
        self.critic2_targ.load_state_dict(self.critic2.state_dict())
        
        model_args['output_transform'] = 'tanh'
        self.actor = General_NN(state_dim, action_dim, model_args, action_max).to(self.device)
        #self.actor.apply(self.init_weights)
        self.actor_targ = General_NN(state_dim, action_dim, model_args, action_max).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.loss_fn = initialize_loss(model_args['loss'])
        
        self.critic_optimizer = initialize_optimizer(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.optimizer_type,
            self.lr_critic
        )
        self.actor_optimizer = initialize_optimizer(
            self.actor.parameters(),
            self.optimizer_type,
            self.lr_actor
        )
        self.buffer = ReplayBufferDeque(self.buffer_size, self.device)
        
        # ignore if not used
        self.ou_noise = OU_Noise(
            size=action_dim, 
            seed=1,
            mu=0.0,
            theta=0.15, 
            sigma=noise_std
        )
        self.ou_noise.reset()

        self.action_low = torch.tensor(self.env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(self.env.action_space.high, dtype=torch.float32)
    
    def init_weights(self, layer):
        """Xaviar Initialization of weights"""
        if(type(layer) == nn.Linear):
          nn.init.xavier_normal_(layer.weight)
          layer.bias.data.fill_(0.01)

    def compute_action(self, state):
        """
        Computes a continuous action with exploratory Gaussian noise.

        Args:
            state (torch.Tensor): The current state, shape (state_dim,) or (c, h, w).

        Returns:
            action (torch.Tensor): A tensor of shape (action_dim,) representing the selected 
                            continuous action with exploration noise added. The action is clipped 
                            to lie within the environment's action space bounds.

        Note: To avoid mismatch error, add batch dimension to state.
        """
        # [obsolete] if discrete, the output of policynn is likely a distribution for actions which u can either sample (stochastic policy) or choose max from (deterministic)
        
        # Action is a tensor on cuda so copy to cpu for env to process it when taking step. This wasn't an issue in dqn implementation bc we used .item(), 
        # but we want action to remain as tensors (fact check) so just move to cpu manually. Check with bipedal walker.
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0)).squeeze(0).cpu() # Add batch dim just for input to model. Detach from grad.
        
            if self.noise_type == 'OU':
                expl_noise = self.ou_noise.sample() # consider expl_noise = torch.tensor(self.ou_noise.sample(), dtype=torch.float32).to(self.device)
            elif self.noise_type == 'Gaussian':
                #expl_noise_std = self.noise_std #0.1 * float(self.env.action_space.high[0])

                expl_noise = torch.randn_like(action) * self.noise_std#expl_noise_std
            else:
                raise NotImplementedError("Noise type not implemented!")
            
            action += expl_noise
        
        #low = torch.tensor(self.env.action_space.low, dtype=action.dtype)
        #high = torch.tensor(self.env.action_space.high, dtype=action.dtype)

        return torch.clamp(action, self.action_low, self.action_high)

    def update_q(self, states, actions, next_states, rewards, terminations):
        """
        Performs a TD3-style update of the Q-networks using target policy smoothing.

        Args:
            states (torch.Tensor): Batch of current states, shape (batch_size, state_dim).
            actions (torch.Tensor): Batch of actions taken, shape (batch_size,).
            next_states (torch.Tensor): Batch of next states, shape (batch_size, state_dim).
            rewards (torch.Tensor): Batch of rewards received, shape (batch_size,).
            terminations (torch.Tensor): Batch of done flags (1 if terminal), shape (batch_size,).

        Returns:
            losses (torch.Tensor): The loss values for critic1 and critic2 respectively.
        """
        # States is of shape torch.Size([batch, features]) but actions is of shape torch.Size([batch]). Convert actions to torch.Size([batch, 1])
    
        q_pred1 = self.critic1(states, actions)
        q_pred2 = self.critic2(states, actions)
        
        if isinstance(self.env, vector.VectorEnv):
            action_clip = float(self.env.action_space.high[0][0])
        else:
            action_clip = float(self.env.action_space.high[0])
        
        targ_actions = self.actor_targ(next_states)

        policy_noise_std = 0.2
        noise = torch.normal(mean=0.0, std=policy_noise_std, size=targ_actions.shape).to(self.device)
        noise_clip = 0.5#action_clip * 0.5 # general case from ChatGPT. Fact check.

        targ_actions_clamped = torch.clamp(
            targ_actions + torch.clamp(
                noise,
                min=-noise_clip,
                max=noise_clip
            ),
            min=-action_clip,
            max=action_clip
        )
        
        with torch.no_grad():
            # change gamma to alg_args
            q_targ = rewards.unsqueeze(-1) + self.gamma * torch.min(
                self.critic1_targ(next_states, targ_actions_clamped),
                self.critic2_targ(next_states, targ_actions_clamped),
            ) * (1 - terminations.unsqueeze(-1))
        
        loss1 = self.loss_fn(q_pred1, q_targ)
        loss2 = self.loss_fn(q_pred2, q_targ)
        #print(loss1, loss2, "ls")
        loss = (loss1 + loss2) / 2 # need the /2 unless we use the same network to produce q1 and q2
        self.critic_optimizer.zero_grad()
        loss.backward()
        #loss1.backward()
        #loss2.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 5)
        #torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 5)
        self.critic_optimizer.step()
        
        return loss#loss1, loss2

    def update_policy(self, states, ep=0):
        """
        Updates the policy network by maximizing the predicted Q-values from critic1.

        Args:
            states (torch.Tensor): Batch of environment states, shape (batch_size, state_dim).

        Returns:
            loss (torch.Tensor): Scalar policy loss used for optimization.
        """
        
        actions = self.actor(states) # may need clipping here
        #if ep > 30:
        #    print("actions", actions, "q", self.critic1(states, actions))
        loss = -torch.mean(self.critic1(states, actions))
        self.actor_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_optimizer.step()
        
        return loss

    def hard_update_target(self, target, source):
        target.load_state_dict(source.state_dict())

    def soft_update_target(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train2(self, episodes):
        
        writer = SummaryWriter(log_dir='runs/' + self.file_pth) # causes overwrite issues

        step = 0
        best_reward = -9999999
        start_time = time.time()

        loss_policy = None
        policy_update_counter = 0
        
        for episode in tqdm(range(episodes), desc="Training episodes"):
            
            state, _ = self.env.reset(seed=self.seed)
            state = preprocess_data(state, self.model_args)
            
            ep_reward = 0
            terminated = False
            truncated = False
            done = terminated or truncated
            while not (done):
                step += 1

                if step < self.learning_starts: # start_steps 
                    action = self.env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float32)
                else:
                    action = self.compute_action(state.to(self.device))
                #print(action)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
                
                next_state = preprocess_data(next_state, self.model_args)
                
                done = terminated or truncated

                self.buffer.add_transition(state, action, next_state, reward, done)
                
                if (step >= self.learning_starts) and (self.buffer.counter >= self.batch_size) and (step % self.train_freq == 0):

                    for _ in range(self.gradient_steps):
                        policy_update_counter += 1

                        states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)
                        
                        loss = self.update_q(states, actions, next_states, rewards, dones)

                        if policy_update_counter % self.policy_delay == 0:
                            loss_policy = self.update_policy(states, episode)
                            self.soft_update_target(self.critic1_targ, self.critic1)
                            self.soft_update_target(self.critic2_targ, self.critic2)
                            self.soft_update_target(self.actor_targ, self.actor)
                        
                state = next_state

                ep_reward += reward

            end_time = time.time()
            elapsed_time = end_time - start_time

            if ep_reward > best_reward:
                best_reward = ep_reward
                print(f"\nBest episode so far! Completed episode {episode} with score {ep_reward}! Elapsed time: {elapsed_time} seconds. Step: {step}.")

            self.actor.save_model(self.file_pth)

            writer.add_scalar("reward", ep_reward, episode)
            if loss_policy is not None:
                writer.add_scalar("loss", loss_policy, episode)
            
        self.env.close()
        writer.flush()
        writer.close()

    def train(self, episodes, num_envs=None):
        writer = SummaryWriter(log_dir='runs/' + self.file_pth) # causes overwrite issues
        
        step = 0
        best_reward = -np.inf
        start_time = time.time()

        loss_policy = None
        policy_update_counter = 0
        
        for episode in tqdm(range(episodes), desc="Training episodes"):
            
            #state = self.env.reset()
            state, _ = self.env.reset(seed=self.seed)
            state = preprocess_data(state, self.model_args)
            
            ep_reward = np.zeros(num_envs) if isinstance(self.env, vector.VectorEnv) else 0

            done = False

            while not np.any(done): # CHANGE np.any(~done)
                step += 1

                if step < self.learning_starts: # start_steps 
                    action = self.env.action_space.sample()
                    action = torch.tensor(action)
                else:
                    action = self.compute_action(state.to(self.device))
                
                #next_state, reward, done, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())

                next_state = preprocess_data(next_state, self.model_args)
                
                if isinstance(self.env, vector.VectorEnv):
                    done = terminated | truncated # for vectorized we need element-wise or since we're working with arrays

                    for i in range(num_envs): #change later to be n_envs
                        self.buffer.add_transition(state[i], action[i], next_state[i], reward[i], done[i]) # try changing to include truncated as well
                        
                else:
                    done = terminated or truncated

                    self.buffer.add_transition(state, action, next_state, reward, done)
                
                if (step >= self.learning_starts) and (self.buffer.counter >= self.batch_size) and (step % self.train_freq == 0):

                    for _ in range(self.gradient_steps):
                        policy_update_counter += 1
                        
                        states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)
                        
                        loss = self.update_q(states, actions, next_states, rewards, dones)
                        
                        if policy_update_counter % self.policy_delay == 0:
                            
                            loss_policy = self.update_policy(states, episode)
                            self.soft_update_target(self.critic1_targ, self.critic1)
                            self.soft_update_target(self.critic2_targ, self.critic2)
                            self.soft_update_target(self.actor_targ, self.actor)
                        
                state = next_state
                ep_reward += reward
            #print(ep_reward)
                
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            
            if isinstance(self.env, vector.VectorEnv):
                if np.mean(ep_reward) > best_reward:
                    best_reward = np.mean(ep_reward)
                    print(f"\nBest episode so far! Completed episode {episode} with score {np.mean(ep_reward)}! Elapsed time: {elapsed_time} seconds. Step: {step}.")
                writer.add_scalar("reward", np.mean(ep_reward), episode)
            else:
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    print(f"\nBest episode so far! Completed episode {episode} with score {ep_reward}! Elapsed time: {elapsed_time} seconds. Step: {step}.")
                writer.add_scalar("reward", ep_reward, episode)
            
            self.actor.save_model(self.file_pth)

            #if loss_policy is not None:
            #    writer.add_scalar("loss", loss_policy, episode)
            
        self.env.close()
        writer.flush()
        writer.close()

    def test(self, model_path):
            self.actor.load_model(model_path)

            state, _ = self.env.reset(seed=self.seed)
            state = preprocess_data(state, self.model_args)

            ep_reward = 0
            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                
                with torch.no_grad():
                    action = self.actor(state.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
                    action = torch.clamp(
                        action,
                        min=float(self.env.action_space.low[0]),
                        max=float(self.env.action_space.high[0])
                    )
                
                step += 1

                next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())

                ep_reward += reward
        
                if (terminated or truncated):
                    print(f"Game end at step: {step}, reward: {ep_reward}!")
                    break
                
                if step % 500 == 0:
                    print(f"Step: {step}!")

                next_state = preprocess_data(next_state, self.model_args)
                state = next_state

class SAC_Agent():
    def __init__(
            self,
            env,
            file_pth,
            model_args,
            lr_critic=1e-3,
            lr_actor=1e-4,
            buffer_size=1_000_000,
            learning_starts=256,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            optimizer_type='Adam',
            seed=None
    ):
        self.env = env
        self.file_pth = file_pth
        self.model_args = model_args
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.optimizer_type = optimizer_type
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            env.action_space.seed(self.seed)

        state, _ = self.env.reset()
        state = preprocess_data(state, model_args)

        action_dim = self.env.action_space.shape[0]

        self.critic1 = Critic(state.shape, action_dim, model_args).to(self.device)
        self.critic2 = Critic(state.shape, action_dim, model_args).to(self.device)
        self.critic1_targ = Critic(state.shape, action_dim, model_args).to(self.device)
        self.critic2_targ =  Critic(state.shape, action_dim, model_args).to(self.device)
        self.actor = General_NN(state.shape, action_dim, model_args).to(self.device)
        self.actor_targ = General_NN(state.shape, action_dim, model_args).to(self.device)

        self.loss_fn = initialize_loss(model_args['name'])

        self.critic_optimizer = initialize_optimizer(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.optimizer_type,
            self.lr_critic
        )
        self.actor_optimizer = initialize_loss(
            self.actor.parameters(),
            self.optimizer_type,
            self.lr_actor
        )

        self.buffer = ReplayBufferDeque(self.buffer_size, self.device, self.seed)

    def entropy(self, log_prob):

        return -self.alpha * log_prob
    
    def sample_action(self, state):
        mu, sd = self.actor(state)
        dist = Normal(mu, sd)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def update_critic(self, states, actions, next_states, rewards, terminations):
        critic_pred1 = self.critic1(states, actions)
        critic_pred2 = self.critic2(states, actions)

        with torch.no_grad():
            mu, sd = self.actor_targ(next_states)
            dist = Normal(mu, sd)
            targ_actions, targ_log_probs = dist.sample(), dist.log_prob()

            target = rewards + (self.gamma * torch.min(
                self.critic1_targ(next_states, targ_actions), 
                self.critic2_targ(next_states, targ_actions)
                ) - self.alpha * targ_log_probs) * (1 - terminations)
        
        loss1 = self.loss_fn(critic_pred1, target)
        loss2 = self.loss_fn(critic_pred2, target)

        self.critic_optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        self.critic_optimizer.step()

        return loss1, loss2
    
    def update_actor(self, states, actions, next_states, rewards, terminations):
        pass
        

    def train(self, episodes):
        
        step = 0
        for ep in tqdm(range(episodes), desc='episodes'):

            state, _ = self.env.reset()
            state = preprocess_data(state)

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, log_prob = self.sample_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.buffer.add_transition(state, action, next_state, reward, terminated)

                if (step < self.buffer.counter):
                    states, actions, next_states, rewards, terminations = self.buffer.sample(self.batch_size)
                    self.update_critic(states, actions, next_states, rewards, terminations)







