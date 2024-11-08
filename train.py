from agent import DQN_Agent, VPG_Agent, TD3_Agent
from utils import compile_args
from env import get_env
#from agent import initialize_agent
from datetime import datetime
import argparse
import shutil
import os
from typing import Type, Union

# This method is implemented here to make agents' docstrings conveniently accessible.
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

if __name__ == "__main__":
    
    # Added command-line config arg for simultaneous hyperparameter tuning.
    parser = argparse.ArgumentParser(description="Run DQN training with a specified config file. Default Atari game is Pong.")
    parser.add_argument('--config', default='configs_pong.yaml', type=str, help="Path to the config YAML file")
    args = parser.parse_args()

    config_path = args.config

    game_args, _, training_args, model_args, parameters = compile_args(config_path) # preprocessor args not used

    # Set default settings. Some may not be used depending on alg. Note: These are settings that may not be preset in config YAML file.
    game_args.setdefault('max_episode_steps', None)
    game_args.setdefault('frame_stack', None) # do for test.py too
    game_args.setdefault('render_mode', 'rgb_array')
    training_args.setdefault('num_envs', None)
    training_args.setdefault('vectorized', False)
    parameters.setdefault('seed', None)

    # Naming model for saves. If save not necessary, name will be '[game]_latest'.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    game_name = game_args['name']
    model_name = f"{game_name}_{timestamp}{training_args['name_tag']}"
    if not training_args['save_model']:
        model_name = game_name + '_latest'

    # Config files are saved with same name as model file in following directory: 'models/[game_name]/'.
    file_pth = os.path.join(game_name, model_name)
    os.makedirs(os.path.join('models', game_name), exist_ok=True)
    shutil.copy(config_path, os.path.join('models', f'{file_pth}.yaml'))

    # Inputs to environment/agent initializations and training.
    alg_type = training_args['alg']
    vectorized = training_args['vectorized']
    num_envs = training_args['num_envs']
    #episodes = training_args['episodes']
    timesteps = training_args['timesteps']
    seed = parameters['seed']
    
    # Initialize game environment. Note that vectorized env is available but the envs are not entirely independent due to shared resets.
    env = get_env(game_args, model_args, seed=seed, vectorize=vectorized, num_envs=num_envs)
    
    agent = initialize_agent(env, alg_type, file_pth, model_args, parameters) 
    #agent = TD3_Agent(env, file_pth, model_args, **parameters) 
    
    agent.train(timesteps, num_envs)

    # consider separate performance memory to sample from the good experiences so that it isnt replaced with bad ones.

    # some sources to read up on for vectorized envs and different reset strategies: https://github.com/Farama-Foundation/Gymnasium/issues/831

    # take a look at the following for accelerated implementations https://github.com/dgriff777/rl_a3c_pytorch
    