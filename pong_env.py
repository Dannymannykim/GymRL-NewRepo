import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FrameStackObservation, AtariPreprocessing
from gymnasium.spaces import Box, Discrete
import numpy as np

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, name=None):
        super().__init__(env)
        self.name = name
        if name == "ALE/Pong-v5":
            # Reduced action space: 2 (up and down)
            self.action_space = gym.spaces.Discrete(2)
            # Mapping: 0 -> 'up' (action 2), 1 -> 'down' (action 3)
            self._action_map = {0: 2, 1: 3}

    def action(self, action_tensor): # tensor to numpy
        # Map the reduced action space to the original actions
        action = action_tensor.item()
        if self.name == "ALE/Pong-v5":
            return self._action_map[action]
        return action
    
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        old_shape = env.observation_space.shape
        framestack, h, w, in_channels = old_shape
        new_shape = (framestack * in_channels, h, w)

        self.observation_space = Box(
            low=env.observation_space.low.min(),  # Minimum value from original space
            high=env.observation_space.high.max(),  # Maximum value from original space
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        framestack, h, w, in_channels = obs.shape
        reshaped_obs = obs.transpose(0, 3, 1, 2).reshape(h, w, -1)  # Combine framestack and in_channels
        return reshaped_obs
    
def get_CustomAtariEnv(model_args, preprocess_args, game_args):
    """
    Function to load custom or customized environments.
    Game Settings:
        - render_mode
        - max_episode_steps
    Preprocess Options:
        - greyscale observations
        - frame stacking
        - convertion to tensor
        - reduced action space (only for "ALE/Pong-v5")
        - (mandatory) resized to (84, 84) 
    Note:
        - For CNNs, a single observation shape is (framestack, h, w, in_channels),
          even when we don't use frame stack. This is to generalize to all cases.
        - For DNNs, the shape is (feature_size,).
    """
    max_episode_steps = game_args.get("max_episode_steps", None)
    render_mode = game_args.get("render_mode", None)
    grayscale_obs = preprocess_args.get("grey-scaled", False)
    stack_size = preprocess_args.get("stack_size", 1)

    env = gym.make(game_args["name"], max_episode_steps=max_episode_steps, render_mode=render_mode) 
    if model_args["nn_type"] == "CNN":
        env = AtariPreprocessing(env, grayscale_obs=grayscale_obs, grayscale_newaxis=True, frame_skip=1) # set frame_skip=1 since original env alrdy frame skips
    
        if (not grayscale_obs and stack_size != 1) or stack_size > 4:
            raise ValueError("Atari games should be grey-scaled for frame stacking since PILImage takes in at most 4 channels! Max frame stack should be 4!") 
        
        env = FrameStackObservation(env, stack_size=stack_size)
        env = ObservationWrapper(env)
    env = ActionWrapper(env, game_args["name"])
    return env



