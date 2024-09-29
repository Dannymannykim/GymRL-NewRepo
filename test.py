from agent import DQN_Agent
#print(torch_directml.is_available())
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from datetime import datetime
import argparse





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""Run DQN agent with its corresponding model file. 
        The corresponding config file will automatically be applied but
        it must have the same name as the model file.
        The defaul game is Pong."""
    )
    parser.add_argument('--model', default='models/pong_latest.pt', type=str, help="Path to the model file (default: 'models/pong_latest.pt')")
    args = parser.parse_args()
    model_path = args.model
    config_path = model_path.replace('.pt', '.yaml')

    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)
    
    # Optional for manually changing certain settings.
    seed = None 
    training_args['seed'] = seed
    game_args['render_mode'] = 'human'
    
    env = get_env(game_args, model_args)#get_CustomAtariEnv(model_args, preprocess_args, game_args)#get_env(model_args)
    file_pth = None
    agent = DQN_Agent(env, file_pth, model_args, optimizer_args, training_args) 

    agent.test(model_path)