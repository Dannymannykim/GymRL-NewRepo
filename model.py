import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_layer_shape

#l1 = conv_layer_shape(5, 4, 0, (84,84))
#l2 = conv_layer_shape(4, 2, 0, l1)
#l3 = conv_layer_shape(3, 1, 0, l2)

def initialize_loss(name):
    """
    Args:
        name (str): The type of loss to use. Can be "MSE" or ....
    """
    if name == "MSE":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {name}")

def initialize_optimizer(params, name, lr, momentum=0):
    """
    Args:
        params (iterable): The parameters of the model to optimize (usually `model.parameters()`).
        name (str): The type of optimizer to use. Can be "Adam" or "SGD".
        lr (float): The learning rate for the optimizer. 
        momentum (float, optional): Momentum factor for SGD.
    """
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {name}")

class General_NN(nn.Module):
    def __init__(self, state_dim, action_dim, model_args):# for conv, state_dim is shape (h,w,channel); for fc, shape (num_features,)

        super().__init__()

        self.nn_type = model_args["nn_type"]
        
        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()

        if self.nn_type == "CNN":
            in_shape = (state_dim[1], state_dim[2])
        
        """
        if self.nn_type == "CNN" and model_args["preprocessing"]["grey-scaled"]:
            if model_args["preprocessing"]["resize"]["enabled"]:
                in_shape = (*model_args["preprocessing"]["resize"]["shape"], 1)
            else:
                in_shape = (*state_dim, 1)
        """
        
        in_channels = state_dim[0] # -1 bc conv shape is (h,w,channel), works for fc shape ; CHANGED: NOW IT'S (channel, h, w)
        #print(state_dim, "S")
        for idx, layer_arg in enumerate(model_args["layer_args"]):

            type = layer_arg["layer_type"]

            if type == "conv" or type == "pooling":
                
                kernel_size = layer_arg["kernel_size"]
                stride = layer_arg["stride"]
                padding  = layer_arg["padding"]
                
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
                    self.conv_layers.add_module(f"relu{idx + 1}", nn.ReLU())

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
                #if in_shape.shape[0] == in_shape.shape[-1]: #checks if input is of shape (feature,) or (h,w,channel)
                #print(in_channels)
                #print(in_channels, in_shape)
                
                out_features = layer_arg["n_out"]
                self.fc_layers.add_module(
                    f"{type}{idx}",
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_features
                    )
                )
                in_channels = out_features
                self.fc_layers.add_module(f"relu{idx + 1}", nn.ReLU())

        # add final       
        self.fc_layers.add_module(
            f"{type}{idx+1}",
            nn.Linear(
                in_features=in_channels,
                out_features=action_dim
            )
        )
        #print(self.fc_layers, "in", action_dim)
        #print(self.conv_layers)
        #print(self.fc_layers)
    
    def forward(self, x):
        x = x / 255
        if self.nn_type == "DNN":
            x = self.fc_layers(x)
        elif  self.nn_type == "CNN":
            x = self.conv_layers(x)
            x = self.fc_layers(x)
        return x
    
class DQN_cnn(nn.Module):
    def __init__(self, action_dim, state_dim=None) -> None:
        super().__init__()

        # Assume grey-scaled frames
        # input shape (batch_size, channels, r, c) = (:, 1, 210, 160)
        # output shape = ([(input_h + 2*padding - kernel_size)/stride] + 1, [(input_w + 2*padding - kernel_size)/stride] + 1)
        
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

