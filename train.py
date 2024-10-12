from agent import DQN_Agent, VPG_Agent
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from agent import initialize_agent
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

    game_args, model_args, optimizer_args, training_args, preprocess_args, alg_args = compile_args(config_path)

    # Set default settings. Some may not be used depending on alg. Note: There are mandatory settings that must be preset in config YAML file.
    game_args.setdefault('max_episode_steps', None)
    game_args.setdefault('render_mode', 'rgb_array')
    alg_args.setdefault('continuous', False)
    training_args.setdefault('batch_size', 64)
    training_args.setdefault('discount', 0.99)
    training_args.setdefault('target_update_method', 'hard')
    training_args.setdefault('step_repeat', 1)
    training_args.setdefault('epsilon', 1)
    training_args.setdefault('epsilon_min', 0.01)
    training_args.setdefault('epsilon_min_ep', training_args['episodes'] * 0.8)
    training_args.setdefault('epsilon_decay', None)

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
    
    env = get_env(game_args, model_args)#get_CustomAtariEnv(game_args, model_args, preprocess_args)

    agent = initialize_agent(env, file_pth, model_args, optimizer_args, training_args, alg_args) 
    
    agent.train()

    # consider separate performance memory to sample from the good experiences so that it isnt replaced with bad ones.