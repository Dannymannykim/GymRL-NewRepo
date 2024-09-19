import gymnasium as gym
import ale_py
import torch
import time
from utils import preprocess_data
from agent import DQN_Agent, DQN_Agent2
#print(torch_directml.is_available())
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation


if __name__ == "__main__":
    
    game = 'pong'
    config_path = "configs_" + game + ".yaml"
    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f'runs/{game}_{timestamp}')
    #model_setup_name = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}' later write name for whether ur doing ddqn, dueling, frame stack, etc

    env = get_env(model_args)

    agent = DQN_Agent(env, model_args, optimizer_args) # DQN2, which is my initial implementation, is faster. The biggest diff is prob the way cuda is sent to device and replay buffer.

    agent.train(training_args)