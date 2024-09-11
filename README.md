### YAML Configuration Documentation

#### `game`
- **`name`** (Mandatory): Name of the game environment (e.g., `CartPole-v1`, `MountainCar-v0`, `ALE/Pong-v5`).
- **`max_episode_steps`** (Optional): Maximum number of steps per episode. 
- **`render_mode`** (Optional): Rendering mode, e.g., `human`, `rgb_array`.

#### `model`
- **`nn_type`** (Mandatory): Neural network type (`DNN`, `CNN`, etc.).
- **`loss`** (Mandatory): Loss function (`MSE`, `CrossEntropy`, etc.).
- **`layer_args`** (Mandatory): Layer configuration. Each layer contains:
  - **`layer_type`**: Type of layer (`fcl`, `conv`, `flatten`).
  - **`n_out`**: Number of output units.

#### `preprocessing`
- **`grey_scale`** (Optional): Boolean to apply grey-scaling. Default: False.
- **`stack_size`** (Optional): Number of frames to stack for frame stacking. Default: 1 (This is set to 1, not None, in order to generalize shape).
- All Atari game envs will undergo resizing to (84, 84).
- No preprocessing required for `DNN`. Use `{}` for default configuration.

#### `optimizer`
- **`name`** (Mandatory): Optimizer name (`SGD`, `Adam`, etc.).
- **`lr`** (Mandatory): Learning rate.

#### `training`
- **`batch_size`** (Mandatory): Batch size for training.
- **`episodes`** (Mandatory): Total number of episodes for training.
- **`discount`** (Mandatory): Discount for Qlearning.
- **`epsilon_min_ep`** (Mandatory): Number of episodes it takes to reach minimum epsilon.
