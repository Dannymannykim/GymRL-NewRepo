### YAML Configuration Documentation

Mandatory and optional arguments. Optional does NOT mean irrelevant; it simply means it is optional
to specify the values because they have default settings.

Also, note that to use the scientific notation e, you must use floats as .yaml will consider it a str.
Ex. 1.0e-4 instead of 1e-4

#### `game`
- **`name`** (Mandatory): Name of the game environment (can be anything). This is simply used for naming model and run files.
- **`version`** (Mandatory): Official name/version of environment (e.g., `CartPole-v1`, `MountainCar-v0`, `ALE/Pong-v5`).
- **`max_episode_steps`** (Optional): Maximum number of steps per episode. [DEFAULT: None (will default to game's own settings; use -1 for no limit at all)]
- **`render_mode`** (Optional): Rendering mode, e.g., `human`, `rgb_array`.

#### `preprocessing`
- **`grey_scale`** (Optional): Boolean to apply grey-scaling. Default: False.
- **`stack_size`** (Optional): Number of frames to stack for frame stacking. Default: 1 (This is set to 1, not None, in order to generalize shape).
- All Atari game envs will undergo resizing to (84, 84).
- No preprocessing required for `DNN`. DEFAULT: `{}`.

#### `training`
- **`alg_type`** (Mandatory): Type of RL agent to train.
- **`episodes`** (Mandatory): Total number of episodes for training.
- **`save_model`** (Mandatory): Whether to save model file or not. If set to false, both configs and model file will be replaced in future runs.
- **`vectorized`** (Optional): Boolean specifying use of vectorized environments for multiprocessing. Default: False.
- **`num_envs`** (Optional/Mandatory if vectorized): Number of environments for multiprocessing. Default: None.
- **`name_tag`** (Mandatory): For naming model and config files. Leave empty string "" if  not used.

#### `model`
- **`continuous`** (Optional): Boolean for whether action space is discrete or cont. Must be false for DQN. Must be true for TD3. [Default: False]
- **`nn_type`** (Mandatory): Neural network type (`DNN`, `CNN`, etc.).
- **`loss`** (Mandatory): Loss function (`MSE`, `CrossEntropy`, etc.).
- **`layer_args`** (Mandatory): Layer configuration. Each layer contains:
  - **`layer_type`**: Type of layer (`fcl`, `conv`, `pooling`, `flatten`).
  - **`n_out`**: Number of output units. For CNNs, this is the number of filters to apply.
  - **`kernel_size`** (Mandatory): Integer of height/width dimension OR tuple of the shape (e.g. 2, (3,5)).
  - **`stride`** (Optional): For CNNs. [DEFAULT: 1]
  - **`padding`** (Optional): For CNNs. Number of padding to apply. [DEFAULT: 0] # GO BACK TO MODEL CLASS AND MAKE SURE
  - **`pooling_type`** (Optional): Type of pooling (`max`, `avg`). For CNNs. 
- **`output_transform`** (Predefined): Transformation applied to final model output. This is helpful when you want squash continuous action values. It is automatically set as `categorical` for discrete VPG and `tanh` for TD3.

#### `parameters`
- **`lr`** (Mandatory/Alg-specific): Learning rate for general networks and methods.
- **`lr_critic`** (Mandatory/Alg-specific): Learning rate specifically for critic networks in actor-critic methods.
- **`lr_actor`** (Mandatory/Alg-specific): Learning rate specifically for actor networks in actor-critic methods.
- **`buffer_size`** (Mandatory): Replay buffer size (# of transitions) for training. For now, this also determines when the agent starts learning (updating weights).
- **`learning_starts`** (Optional): How many steps after to start updating weights. [DEFAULT: 1]
- **`batch_size`** (Optional): Batch size for training. [DEFAULT: 64]
- **`tau`** (Optional): the soft update coefficient (“Polyak update”, between 0 and 1) [Default: ]
- **`gamma`** (Optional): Discount for Qlearning. [DEFAULT: 0.99]
- **`optimizer_type`** (Mandatory): Optimizer name (`SGD`, `Adam`, etc.).
- **`train_freq`** (Optional): Update the model every train_freq steps.
- **`gradient_steps`** (Optional): How many gradient steps to do after each rollout
- **`noise_type`** (Optional/Alg-specific): Type of exploratory noise for computing continuous actions in cont. actor-critic methods (e.g., `Gaussian`, `OU`). [Default: Gaussian]
- **`policy_delay`** (Optional): Policy and target networks will only be updated once every policy_delay steps per training steps. The Q values will be updated policy_delay more often (update every training step).
- **`seed`** (Optional): Seed to control randomness in environment and training. Mainly for hyperparameter tuning. [DEFAULT: None]
- **`epsilon`** (Optional): Parameter for exploration-exploitation tradeoff. [DEFAULT: 1]
- **`epsilon_min`** (Optional): Minimum epsilon. [DEFAULT: 0.01]
- **`epsilon_min_ep`** (Optional): For exponential epsilon decay. Episode at which epsilon is decayed to the minimum value. [DEFAULT: episodes * 0.8]
- **`epsilon_decay`** (Optional): For linear epsilon decay. [DEFAULT: exponential decay]
- **`step_repeat`** (Optional): How many steps to process at a time. [DEFAULT: 1]
- **`target_update_interval`** (Mandatory/Alg-specific): Number of steps until target network is updated.
- **`target_update_method`** (Optional): Method to update target method (`hard`, `soft`). [DEFAULT: `hard`]