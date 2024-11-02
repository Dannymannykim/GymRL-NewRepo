import torch
import torch.nn as nn
from utils import conv_layer_shape
import os
from torch.distributions import Categorical

def initialize_loss(name):
    """
    Args:
        - name (str): The type of loss to use. Can be "MSE" or ....
    Returns:
        - torch.nn.Module: The initialized loss function instance.
    """
    if name == "MSE":
        return nn.MSELoss()
    elif name == "Huber":
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unsupported loss type: {name}")

def initialize_optimizer(params, name, lr, momentum=0):
    """
    Args:
        - params (iterable): The parameters of the model to optimize (usually `model.parameters()`).
        - name (str): The type of optimizer to use. Can be "Adam" or "SGD".
        - lr (float): The learning rate for the optimizer. 
        - momentum (float, optional): Momentum factor for SGD.
    Returns:
        - torch.optim.Optimizer: The initialized optimizer instance.
    """
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {name}")
    
def initialize_activation(activation):
    """
    Args:
        - activation (str): The type of activation (e.g., relu, leaky_relue). Default: Relu.
    Returns:
        - torch.nn.Module: The initialized activation function instance.
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation type: {activation}")
    
def initialize_output_transform(name):
    """
    For applying transformation to network output. Needed for different algorithms.
    E.g. VPG requires categorical, TD3 with certain envs may improve with tanh.
    """
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'categorical':
        return lambda x: Categorical(logits=x)
    else:
        raise ValueError(f"Unknown output transform: {name}")

class General_NN(nn.Module):
    """
    A general-purpose Pytorch neural network model that can handle both Deep Neural Networks (DNNs) 
    and Convolutional Neural Networks (CNNs) for reinforcement learning tasks. It is designed 
    to take in either flat feature vectors (for DNN) or image-like data (for CNN), and output 
    an action or Q-values based on the task at hand.
    
    Args:
        - state_dim (tuple): For DNN, shape is (features,). For CNN, shape is (channels, h, w).
        - action_dim (int): The number of output nodes corresponding to the action space's dimension.
        - model_args (dict): Arguments for network architecture, such as layer sizes, activation 
          functions, etc. (see README.md).
        - action_max (int): Specifically for soft-actor methods. (FACT CHECK)

    [Note: For q-networks in Soft-Actor critic methods, use Critic class instead.]
    """
    def __init__(self, state_dim, action_dim, model_args, action_max=None):
        super().__init__()

        self.nn_type = model_args["nn_type"]
        self.model_args = model_args
        self.action_max = action_max

        self.activation = initialize_activation(model_args.get('activation', 'relu'))
        
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()

        if self.nn_type == "CNN":
            in_shape = (state_dim[1], state_dim[2])
        
        in_channels = state_dim[0]
        
        for idx, layer_arg in enumerate(model_args["layer_args"]):
            type = layer_arg["layer_type"]

            if type == "conv" or type == "pooling":
                kernel_size = layer_arg['kernel_size']
                stride = layer_arg.get('stride', 1)
                padding  = layer_arg.get('padding', 0)
                
                if type == "conv":
                    out_channels = layer_arg["n_out"]
                    self.conv_layers.add_module(
                        f"{type}{idx}",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
                    self.conv_layers.add_module(f"{model_args.get('activation', 'relu')}{idx + 1}", self.activation)

                if type == "pooling": 
                    if layer_arg["pooling_type"] == "max":
                        self.conv_layers.add_module(
                            f"{type}{idx}",
                            nn.MaxPool2d(
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )

                    elif layer_arg["pooling_type"] == "avg":
                        self.conv_layers.add_module(
                            f"{type}{idx}",
                            nn.AvgPool2d(
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
                
                in_channels = out_channels
                in_shape = conv_layer_shape(kernel_size, stride, padding, (in_shape[0], in_shape[1]))
                
            elif layer_arg["layer_type"] == "flatten":
                self.conv_layers.add_module("flatten", nn.Flatten())
            
                in_channels=in_channels * in_shape[0] * in_shape[1]
            
            elif layer_arg["layer_type"] == "fcl":
                out_features = layer_arg["n_out"]

                self.fc_layers.add_module(
                    f"{type}{idx}",
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_features
                    )
                )

                in_channels = out_features
                self.fc_layers.add_module(f"{model_args.get('activation', 'relu')}{idx + 1}", self.activation)

        # add final output layer
        self.fc_layers.add_module(
            f"{type}{idx+1}",
            nn.Linear(
                in_features=in_channels,
                out_features=action_dim
            )
        )
            
    def forward(self, x):
        """
        Performs forward pass. Inputs of CNNs are normalized.

        The following are outputs for each type of task:
            - Continuous Q-network: Use Critic class instead.
            - Discrete Q-network: Outputs Q-values for each discrete action in the action space. 
              The output is a vector where each element corresponds to the Q-value of a specific action.
            - Continuous Policy-network: Outputs a parameter of the policy, such as the mean. If policy 
              is stochastic, the parameter will be used to sample an action from a distribution. Otherwise, 
              it will be used as the continuous action directly.
            - Discrete Policy-network: Outputs a probability distribution over discrete actions. The output 
              is a vector where each element corresponds to the probability of selecting a particular action.
        """
        if self.nn_type == "DNN":
            x = self.fc_layers(x)
            #print(self.fc_layers, 'd')
        elif self.nn_type == "CNN":
            x = x / 255
            x = self.conv_layers(x)
            x = self.fc_layers(x)

        output_transform = self.model_args.get('output_transform', None)

        if output_transform is not None:
            if output_transform == 'Categorical' and self.model_args['continuous']:
                raise ValueError("Categorical distribution cannot be used for continuous action spaces!")
            elif output_transform == 'tanh' and not self.model_args['continuous']:
                raise ValueError("Tanh activation should not be used for discrete action spaces!")
            
            transform = initialize_output_transform(output_transform)

            x = transform(x)

            # Scale only for continuous actions + tanh (actor-critic methods)
            if output_transform == 'tanh' and self.model_args['continuous']:
                x = x * self.action_max

        return x
    
    def save_model(self, file_pth):
        """
        Creates a directory 'model' and saves model.
        """
        if not os.path.exists('models'):
            os.makedirs('models')

        torch.save(self.state_dict(), os.path.join('models', file_pth + '.pt'))

    def load_model(self, file):
        """
        Loads model from full file path.
        """
        try:
            self.load_state_dict(torch.load(file))
            self.eval() # for dropout or batch normalization
            print(f"Loaded weights from {file}!")
        except FileNotFoundError:
            print(f"No weights file found at at {file}!")


class Critic(nn.Module):
    """
    A separate architecture for Actor-Critic methods, specifically designed for 
    Q-networks in algorithms like DDPG and TD3. These algorithms use a Q-network 
    that takes both the state and the action as input, in contrast to traditional 
    Q-networks, which only take the state as input and output Q-values for all 
    possible actions. This method is particularly necessary for continuous action 
    spaces, where actions must be explicitly provided as input.

    In DDPG and TD3, the action dimension is incorporated into the network's 
    architecture, typically added to the second layer of a Deep Neural Network (DNN) 
    or before the fully connected layers in a Convolutional Neural Network (CNN).

    Args:
        - state_dim (tuple): For DNN, shape is (features,). For CNN, shape is (channels, h, w).
        - action_dim (int): The number of output nodes corresponding to the action space's dimension.
        - model_args (dict): Arguments for network architecture, such as layer sizes, activation functions, etc. (see README.md).
        - alg_args (dict): Arguments for the algorithm (see README.md).
    """
    def __init__(self, state_dim, action_dim, model_args):
        super().__init__()

        self.model_args = model_args

        self.nn_type = model_args["nn_type"]
        self.activation = initialize_activation(model_args.get('activation', 'relu'))

        if self.nn_type == "CNN":
            in_shape = (state_dim[1], state_dim[2])
        
        in_channels = state_dim[0]

        if model_args['continuous'] and self.nn_type == 'DNN':
            if len(model_args["layer_args"]) < 2:
                raise NotImplementedError("Missing second layer. DDPG and TD3 adds action dim to second layer of Q network.")
            
            out_channels = model_args['layer_args'][0]['n_out']
            self.fcl1 = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                self.activation
            )
            in_channels = out_channels + action_dim

        self.fc_layers = nn.Sequential()
        self.conv_layers = nn.Sequential()

        for idx, layer_arg in enumerate(model_args["layer_args"][1:]):

            type = layer_arg["layer_type"]

            if type == "conv" or type == "pooling":
                
                kernel_size = layer_arg['kernel_size']
                stride = layer_arg.get('stride', 1)
                padding  = layer_arg.get('padding', 0)
                
                if type == "conv":
                    out_channels = layer_arg["n_out"]
                    self.conv_layers.add_module(
                        f"{type}{idx}",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
                    self.conv_layers.add_module(f"{model_args.get('activation', 'relu')}{idx + 1}", self.activation)

                if type == "pooling": 
                    if layer_arg["pooling_type"] == "max":
                        self.conv_layers.add_module(
                            f"{type}{idx}",
                            nn.MaxPool2d(
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )

                    elif layer_arg["pooling_type"] == "avg":
                        self.conv_layers.add_module(
                            f"{type}{idx}",
                            nn.AvgPool2d(
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
                
                in_channels = out_channels
                in_shape = conv_layer_shape(kernel_size, stride, padding, (in_shape[0], in_shape[1]))
                
            elif layer_arg["layer_type"] == "flatten":
                self.conv_layers.add_module("flatten", nn.Flatten())
            
                in_channels=in_channels * in_shape[0] * in_shape[1]

                if self.model_args['continuous']:
                    in_channels += action_dim
            
            elif layer_arg["layer_type"] == "fcl":
                out_features = layer_arg["n_out"]
                
                self.fc_layers.add_module(
                    f"{type}{idx}",
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_features
                    )
                )
                in_channels = out_features
                self.fc_layers.add_module(f"{model_args.get('activation', 'relu')}{idx + 1}", self.activation)

        # add final output layer
        self.fc_layers.add_module(
            f"{type}{idx+1}",
            nn.Linear(
                in_features=in_channels,
                out_features=action_dim
            )
        )

    def forward(self, x, a=None):
        """
        Performs forward pass. Inputs of CNNs are normalized.

        Passes both state and action as input. The action is passed into
        the second layer of a DNN or before the fc layer of a CNN.

        Note: Make sure x and a are the same dimension. Specifically, action
        input a should be passed in with an extra dimension. DO NOT do this 
        in the foward method as there will be errors due to target actions 
        already having an extra dimension.
        """
        is_continuous = self.model_args['continuous']

        if self.nn_type == 'DNN':
            if is_continuous:
                x = self.fcl1(x)
                #print(x.shape, a.shape)
                x = torch.cat((x, a), dim=1) 
            x = self.fc_layers(x)

        elif self.nn_type == 'CNN':
            x = x / 255
            x = self.conv_layers(x)
            if is_continuous:   
                x = torch.cat([x, a], dim=1) # maybe add dim to 2nd layer of fcl for CNN as well
            x = self.fc_layers(x)
        
        return x


class DQN_cnn(nn.Module):
    """
    [OBSOLETE]
    """

    def __init__(self, action_dim, state_dim=None) -> None:
        super().__init__()
        
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),  #  (:, 16, 105.5 -> 105, 80.5 -> 80)
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # (:, 32, 53, 40)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # (:, 64, 27, 20)
            nn.ReLU(),
            nn.Flatten()  # (batch, 34560)
        )
        
        self.fcl = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # Adjust dimensions based on input size and conv layers
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, x):
        x = self.convolution(x)
        x = self.fcl(x)
        return x
    
class DQN_dnn(nn.Module):
    """
    [OBSOLETE]
    """

    def __init__(self, action_dim, state_dim) -> None:
        super().__init__()
        
        self.fcl = nn.Sequential(
            nn.Linear(state_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        x = self.fcl(x)
        return x

