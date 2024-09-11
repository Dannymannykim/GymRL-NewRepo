import gymnasium as gym
import ale_py
import torch
import time
from utils import preprocess_data
from dqn_agent import DQN_Agent
import torch_directml  # Import torch-directml
#print(torch_directml.is_available())
from utils import compile_args
from pong_env import get_CustomAtariEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# gymnasium state shape: (r, c, channels)
# input pytorch shape: batch size, channels, r, c

if __name__ == "__main__":
    writer = SummaryWriter() # by default, this will create new folders. 
    # SummaryWriter('runs/folder_name') seems to be not overriding but instead writing over the previous result

    config_path = "configs_cartpole.yaml"

    game_args, model_args, optimizer_args, training_args, preprocess_args = compile_args(config_path)

    #device = torch_directml.device() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # you only have to specify device manually for pytorch

    env = get_CustomAtariEnv(model_args, preprocess_args, game_args)
    
    action_dim = env.action_space.n #np.array([2, 3]) # atari pong has 6 action spaces but we really only need 2 and 3
    state_dim = env.observation_space.shape
    agent = DQN_Agent(device, action_dim, state_dim, model_args, optimizer_args, training_args)
    
    episodes = training_args["episodes"]

    start_time = time.time()
    step = 0
    best_reward = -9999999

    for episode in tqdm(range(episodes), desc="Training episodes"):#for episode in range(episodes):
        state, _ = env.reset()
        
        state = preprocess_data(device, state, model_args)
        
        ep_reward = 0
        while True:
            step += 1

            action = agent.choose_action(env, state, device) 
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward

            reward = torch.tensor(reward, dtype=torch.float, device=device)
            terminated = torch.tensor(terminated, dtype=torch.float, device=device)           
        
            next_state = preprocess_data(device, next_state, model_args)

            agent.replay_buffer.add_state((state, next_state, reward, action, terminated))
            
            loss = agent.update_weights()
            
            state = next_state

            if step % agent.target_update_interval == 0:
                agent.update_target()

            if terminated or truncated:
                end_time = time.time()
                elapsed_time = end_time - start_time
                break
        
        if episode % 1 == 0 and episode > agent.batch_size: # change so that it averages over that 100 interval
            writer.add_scalar("reward", ep_reward, episode)
            writer.add_scalar("loss", loss, episode)

        if episode % 1000 == 0:
            print(episode, best_reward)
            
        if ep_reward > best_reward:
            best_reward = ep_reward
            print("episode", episode, "reward", ep_reward, "epsilon", agent.epsilon, "best", best_reward)
        
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        writer.add_scalar("Epsilon vs. Episodes", agent.epsilon, episode)
            
    env.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    writer.flush()
    writer.close()