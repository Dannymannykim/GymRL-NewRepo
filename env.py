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

def get_env(game_args, model_args, seed=None, vectorize=False, num_envs=None):
    """
    Note:
        - For CNNs, a single observation shape is (h, w, in_channels).
          With frame stacks, the shape is (frame_stacks, h, w, in_channels).
          We want shape (h, w, in_channels * frame_stacks).
        - For DNNs, the shape is (feature_size,).
        - Reshaping for pytorch shape (in_channels, h, w) will be done separately.
    
    Maybe set defaults here.
    """
    game_id = game_args['version']
    render_mode = game_args['render_mode']

    max_ep_steps = game_args['max_episode_steps']
    frame_stack = game_args['frame_stack']

    policy_type = model_args["nn_type"]

    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    wrapper = (lambda env: GrayscaleObservation(ResizeObservation(env, (64, 64)), keep_dim=True)) if model_args["nn_type"] == "CNN" else None
    if vectorize:
        #env = make_vec_env(
        #    env_id=game_args['version'],
        #    n_envs=3,
        #    vec_env_cls=SubprocVecEnv,
        #    wrapper_class=wrapper,
        #    seed=seed
        #) # this gives shape (24,) for observation_space and sampling action gives action for one env.
        env = gym.make_vec(game_args['version'], num_envs=num_envs, vectorization_mode="async", render_mode=game_args['render_mode'])
        
    else:
        env = gym.make(game_id, max_episode_steps=max_ep_steps, render_mode=render_mode)
        
        if policy_type == "CNN":
            env = ResizeObservation(env, (84, 84))  
            env = GrayscaleObservation(env, keep_dim=True)

            if frame_stack is not None:
                env = FrameStackObservation(env, stack_size=frame_stack)
                env = ObservationWrapper(env) # check if this breaks seeding
    
    return env

