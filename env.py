import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FrameStackObservation, AtariPreprocessing
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.spaces import Box, Discrete
import numpy as np
import ale_py

class ObservationWrapper(gym.ObservationWrapper):
    """
    Directly incorporates frame stacking and shape
    conversion for Pytorch into the environment itself.
    """

    def __init__(self, env):
        super().__init__(env)

        old_shape = env.observation_space.shape
        framestack, h, w, in_channels = old_shape
        new_shape = (h, w, framestack * in_channels)

        self.observation_space = Box(
            low=env.observation_space.low.min(),  # Minimum value from original space
            high=env.observation_space.high.max(),  # Maximum value from original space
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        framestack, h, w, in_channels = obs.shape
        reshaped_obs = obs.reshape(h, w, -1)  # Combine framestack and in_channels
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
        - resize to (x, y) 
    Note:
        - For CNNs, a single observation shape is (h, w, in_channels).
          With frame stacks, the shape is (frame_stacks, h, w, in_channels).
          We want shape (h, w, in_channels * frame_stacks).
        - For DNNs, the shape is (feature_size,).
        - Reshaping for pytorch shape (in_channels, h, w) will be done separately.
    """

    max_episode_steps = game_args.get("max_episode_steps", None)
    render_mode = game_args.get("render_mode", None)
    grayscale_obs = preprocess_args.get("grey-scaled", False)
    stack_size = preprocess_args.get("stack_size", 1)

    env = gym.make(game_args["name"], max_episode_steps=max_episode_steps, render_mode=render_mode) 
    if model_args["nn_type"] == "CNN":
        env = AtariPreprocessing(env, grayscale_obs=grayscale_obs, grayscale_newaxis=True, frame_skip=1, screen_size=64) # set frame_skip=1 since original env alrdy frame skips
    
        if (not grayscale_obs and stack_size != 1) or stack_size > 4:
            raise ValueError("Atari games should be grey-scaled for frame stacking since PILImage takes in at most 4 channels! Max frame stack should be 4!") 
        elif stack_size > 1:
            env = FrameStackObservation(env, stack_size=stack_size)
            env = ObservationWrapper(env)
            #print("SS")
    #print(env.observation_space.shape, "S")
    #raise ImportError
    return env

def get_env(model_args, preprocess_args=None, game_args=None):

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    env = ResizeObservation(env, (64, 64))
    
    if model_args["nn_type"] == "CNN":

        env = GrayscaleObservation(env, keep_dim=True)

    return env

