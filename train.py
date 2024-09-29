from agent import DQN_Agent
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from datetime import datetime
import argparse
import shutil
import os

# ALL PARAMETERS/ARGUMENTS ARE SET IN CONFIG FILE! 

if __name__ == "__main__":
    
    # Added command-line config arg for simultaneous hyperparameter tuning.
    parser = argparse.ArgumentParser(description="Run DQN training with a specified config file. Default Atari game is Pong.")
    parser.add_argument('--config', default='configs_pong.yaml', type=str, help="Path to the config YAML file")
    args = parser.parse_args()

    config_path = args.config

    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)

    # Naming model for saves. If save not necessary, name will be '[game]_latest'.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    game_name = game_args['name']
    model_name = f"{game_name}_{timestamp}{model_args['name_tag']}"
    if not training_args['save_model']:
        model_name = game_name + '_latest'

    # Config files are saved with same name as model file in following directory: 'models/[game_name]/'.
    file_pth = os.path.join(game_name, model_name)
    os.makedirs(os.path.join('models', game_name), exist_ok=True)
    shutil.copy(config_path, os.path.join('models', f'{file_pth}.yaml'))
    
    env = get_env(game_args, model_args)#get_CustomAtariEnv(game_args, model_args, preprocess_args)#get_env(model_args)

    agent = DQN_Agent(env, file_pth, model_args, optimizer_args, training_args) 

    agent.train()

    # consider separate performance memory to sample from the good experiences so that it isnt replaced with bad ones.