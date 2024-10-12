import torch
import torch.nn as nn
from utils import conv_layer_shape
import os
from torch.distributions import Categorical

def initialize_loss(name):
    """
    Args:
        - name (str): The type of loss to use. Can be "MSE" or ....
    """

    if name == "MSE":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {name}")

def initialize_optimizer(params, name, lr, momentum=0):
    """
    Args:
        - params (iterable): The parameters of the model to optimize (usually `model.parameters()`).
        - name (str): The type of optimizer to use. Can be "Adam" or "SGD".
        - lr (float): The learning rate for the optimizer. 
        - momentum (float, optional): Momentum factor for SGD.
    """

    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {name}")
    
def initialize_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation type: {activation}")
    
    

class General_NN(nn.Module):

    def __init__(self, state_dim, action_dim, model_args, alg_args):
        #self.count = 0
        super().__init__()

        self.nn_type = model_args["nn_type"]
        self.alg_args = alg_args
        self.activation = initialize_activation(model_args.get('activation', 'relu'))
        
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()

        if self.nn_type == "CNN":
            in_shape = (state_dim[1], state_dim[2])
        
        in_channels = state_dim[0] # -1 bc conv shape is (h,w,channel), works for fc shape ; CHANGED: NOW IT'S (channel, h, w)
        
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
        
        if self.nn_type == "DNN":
            x = self.fc_layers(x)

        elif self.nn_type == "CNN":
            x = x / 255
            x = self.conv_layers(x)
            x = self.fc_layers(x)
        
        if self.alg_args['type'] == 'VPG':
            
            if self.alg_args['continuous']:
                pass
            else:
                x = Categorical(logits=x) # Note: returns dist NOT tensor
        
        return x
    
    def save_model(self, file_pth):
        """
        Creates a directory 'model' and saves model.
        """
        if not os.path.exists('models'):
            os.makedirs('models')

        torch.save(self.state_dict(), os.path.join('models', file_pth + '.pt'))

    def load_model(self, file):
        try:
            self.load_state_dict(torch.load(file))
            self.eval() # for dropout or batch normalization
            print(f"Loaded weights from {file}!")
        except FileNotFoundError:
            print(f"No weights file found at at {file}!")

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

