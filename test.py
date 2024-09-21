from agent import DQN_Agent
#print(torch_directml.is_available())
from utils import compile_args
from env import get_CustomAtariEnv, get_env
from datetime import datetime






if __name__ == "__main__":
    
    game = 'pong'
    config_path = "configs_" + game + ".yaml"
    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)

    model_file = 'models/pong_bestsofar2.pt'
    
    env = get_env(model_args)#get_CustomAtariEnv(model_args, preprocess_args, game_args)#get_env(model_args)

    agent = DQN_Agent(env, model_args, optimizer_args) 

    agent.test(model_file)