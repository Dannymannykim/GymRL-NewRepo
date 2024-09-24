from agent import DQN_Agent
#print(torch_directml.is_available())
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from datetime import datetime
import argparse





if __name__ == "__main__":
    seed = 0

    parser = argparse.ArgumentParser(
        description="""Run DQN agent with its corresponding config file. 
        The config file must have the same name as the model file.
        The defaul game is Pong."""
    )
    parser.add_argument('--config', default='models/pong_latest.yaml', type=str, help="Path to the config YAML file (default: 'models/pong_latest.yaml)")
    parser.add_argument('--model', default='models/pong_latest.pt', type=str, help="Path to the model file (default: 'models/pong_latest.pt')")
    args = parser.parse_args()
    config_path = args.config  
    model_path = args.model

    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)
    
    env = get_env(game_args, model_args)#get_CustomAtariEnv(model_args, preprocess_args, game_args)#get_env(model_args)

    agent = DQN_Agent(env, model_args, optimizer_args, seed) 

    agent.test(model_path)