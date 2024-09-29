### YAML Configuration Documentation

Mandatory and optional arguments. Optional does NOT mean irrelevant; it simply means it is optional
to specify the values because they have default settings.

#### `game`
- **`name`** (Mandatory): Name of the game environment (can be anything). This is simply used for naming model and run files.
- **`version`** (Mandatory): Official name/version of environment (e.g., `CartPole-v1`, `MountainCar-v0`, `ALE/Pong-v5`).
- **`max_episode_steps`** (Optional): Maximum number of steps per episode. [DEFAULT: None (will default to game's own settings; use -1 for no limit at all)]
- **`render_mode`** (Optional): Rendering mode, e.g., `human`, `rgb_array`.

#### `model`
- **`name_tag`** (Mandatory): For naming model and config files. Leave empty string "" if  not used.
- **`nn_type`** (Mandatory): Neural network type (`DNN`, `CNN`, etc.).
- **`loss`** (Mandatory): Loss function (`MSE`, `CrossEntropy`, etc.).
- **`layer_args`** (Mandatory): Layer configuration. Each layer contains:
  - **`layer_type`**: Type of layer (`fcl`, `conv`, `pooling`, `flatten`).
  - **`n_out`**: Number of output units. For CNNs, this is the number of filters to apply.
  - **`kernel_size`** (Mandatory): Integer of height/width dimension OR tuple of the shape (e.g. 2, (3,5)).
  - **`stride`** (Optional): For CNNs. [DEFAULT: 1]
  - **`padding`** (Optional): For CNNs. Number of padding to apply. [DEFAULT: 0] # GO BACK TO MODEL CLASS AND MAKE SURE
  - **`pooling_type`** (Optional): Type of pooling (`max`, `avg`). For CNNs. 

#### `preprocessing`
- **`grey_scale`** (Optional): Boolean to apply grey-scaling. Default: False.
- **`stack_size`** (Optional): Number of frames to stack for frame stacking. Default: 1 (This is set to 1, not None, in order to generalize shape).
- All Atari game envs will undergo resizing to (84, 84).
- No preprocessing required for `DNN`. DEFAULT: `{}`.

#### `optimizer`
- **`name`** (Mandatory): Optimizer name (`SGD`, `Adam`, etc.).
- **`lr`** (Mandatory): Learning rate.

#### `training`
- **`seed`** (Optional): Seed to control randomness in environment and training. Mainly for hyperparameter tuning. [DEFAULT: None]
- **`save_model`** (Mandatory): Whether to save model file or not. If set to false, both configs and model file will be replaced in future runs.
- **`batch_size`** (Optional): Batch size for training. [DEFAULT: 64]
- **`buffer_size`** (Mandatory): Replay buffer size (# of transitions) for training. For now, this also determines when the agent starts learning (updating weights).
- **`episodes`** (Mandatory): Total number of episodes for training.
- **`discount`** (Optional): Discount for Qlearning. [DEFAULT: 0.99]
- **`epsilon`** (Optional): Parameter for exploration-exploitation tradeoff. [DEFAULT: 1]
- **`epsilon_min`** (Optional): Minimum epsilon. [DEFAULT: 0.01]
- **`epsilon_min_ep`** (Optional): For exponential epsilon decay. Episode at which epsilon is decayed to the minimum value. [DEFAULT: episodes * 0.8]
- **`epsilon_decay`** (Optional): For linear epsilon decay. [DEFAULT: exponential decay]
- **`target_update_interval`** (Mandatory): Number of steps until target network is updated.
- **`target_update_method`** (Optional): Method to update target method (`hard`, `soft`). [DEFAULT: `hard`]
- **`step_repeat`** (Optional): How many steps to process at a time. [DEFAULT: 1]
