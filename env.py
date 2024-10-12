import gymnasium as gym
from gymnasium.wrappers import TimeLimit, FrameStackObservation, AtariPreprocessing
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.spaces import Box, Discrete
import numpy as np
import ale_py

class ObservationWrapper(gym.ObservationWrapper):
    """
    [OBSOLETE]

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
        
# Some wrappers applied to the environment may override the original
# reset method and exlude the seed attribute. In such cases, simply
# create a custom wrapper of the top-most wrapper (last wrapper applied
# that overrides the reset method without seed). Below are some examples.

# Also note that some overrides accept seed but enforce keyword-only arguments,
# meaning you have to specify 'seed', e.g. env.reset(seed=seed). This happens
# in implementations that use *, e.g. def reset(*, seed=None). If a function 
# has * in its parameter list, any argument that appears after * must be passed 
# as a keyword argument.

# Observation wrapper does not take in seed, so we need to override.
# Note that we only need to override the last wrapper applied that
# overrides the reset method. ResizeObservation and GreyscaleObservation
# themselves don't override reset, but their parent wrapper does.
class CustomResizeObservation(ResizeObservation): 
    def reset(self, seed=None, **kwargs):
        # Pass seed to the underlying environment if necessary
        return super().reset(seed=seed, **kwargs)

class CustomGrayscaleObservation(GrayscaleObservation):
    def reset(self, seed=None, **kwargs):
        # Pass seed to the underlying environment if necessary
        return super().reset(seed=seed, **kwargs)

def get_CustomAtariEnv(game_args, model_args, preprocess_args):
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

    env = gym.make(game_args["version"], max_episode_steps=max_episode_steps, render_mode=render_mode) 
    if model_args["nn_type"] == "CNN":
        env = AtariPreprocessing(env, grayscale_obs=grayscale_obs, grayscale_newaxis=True, frame_skip=1, screen_size=64) # set frame_skip=1 since original env alrdy frame skips
    
        if (not grayscale_obs and stack_size != 1) or stack_size > 4:
            raise ValueError("Atari games should be grey-scaled for frame stacking since PILImage takes in at most 4 channels! Max frame stack should be 4!") 
        elif stack_size > 1:
            env = FrameStackObservation(env, stack_size=stack_size)
            env = ObservationWrapper(env)
            
    return env

def get_env(game_args, model_args):

    env = gym.make(game_args['version'], max_episode_steps=game_args['max_episode_steps'], render_mode=game_args['render_mode'])
    
    if model_args["nn_type"] == "CNN":

        env = ResizeObservation(env, (64, 64))

        env = GrayscaleObservation(env, keep_dim=True)

    return env

