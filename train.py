from agent import DQN_Agent
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from datetime import datetime
import argparse
import shutil
import os

if __name__ == "__main__":
    
    seed = 0 # set seed
    
    # Added command-line config arg for simultaneous hyperparameter tuning.
    parser = argparse.ArgumentParser(description="Run DQN training with a specified config file. Default Atari game is Pong.")
    parser.add_argument('--config', default='configs_pong.yaml', type=str, help="Path to the config YAML file")
    args = parser.parse_args()

    config_path = args.config

    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)

    #saved_hyperparameters = {**model_args, **training_args} 
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"{game_args['name']}_{timestamp}_{model_args['name_tag']}"
    if not training_args['save_model']:
        model_name = game_args['name'] + "_latest"

    if not os.path.exists('models'):
        os.makedirs('models')
    shutil.copy(config_path, f"models/{model_name}.yaml")
    
    env = get_env(game_args, model_args)#get_CustomAtariEnv(game_args, model_args, preprocess_args)#get_env(model_args)

    agent = DQN_Agent(env, model_name, model_args, optimizer_args, training_args, seed) 

    agent.train()