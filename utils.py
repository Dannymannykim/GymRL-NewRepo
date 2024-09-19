import torch
import cv2
from torchvision import transforms
import os
import torch.nn as nn
import math
import yaml
import numpy as np

def create_layer_args(
        layer_type, 
        n_out=None, 
        kernel_shape=None, 
        stride=None, 
        padding=None, 
        pool_type=None,
):
    """
    (OBSOLETE; We're using .yaml now)

    Create a dictionary of arguments for defining a layer (exclude output layer). 

    Parameters:
        layer_type (str): The type of layer to create. Available options are:
            - "fcl": Fully connected layer
            - "conv": Convolutional layer
            - "pool": Pooling layer (e.g., max-pooling or average-pooling)
            - "flatten": Flattening layer (for after conv)

    Returns:
        dict: A dictionary containing the layer configuration.
    """
    return {
        "layer_type": layer_type,
        "n_out": n_out,
        "kernel_shape": kernel_shape,
        "stride": stride,
        "padding": padding,
        "pooling_type": pool_type,
    }

def conv_layer_shape(kernel_size, stride, padding, in_shape):
    
    shape = (
        math.floor(((in_shape[0] + 2*padding - kernel_size) / stride) + 1), 
        math.floor(((in_shape[1] + 2*padding - kernel_size) / stride) + 1)
    )
    #print(shape)
    return shape

def preprocess_data(device, frame, model_args=None):
    """
    Preprocesses an input RGB frame.

    Args:
        frame (numpy.ndarray): RGB frame of shape (H, W, C).

    Returns:
        torch.Tensor: Preprocessed frame of shape (C, H, W).

    Steps:
        1. Convert the frame (NumPy array) to a PIL image of shape (C, H, W).
        2. [Ignore] Convert to grayscale, resulting in shape (1, H, W).
        3. [Ignore] Optionally resize to (1, 84, 84). 
           (The 84x84 size is a standard for training Atari-based RL agents.)
        4. Convert the frame to a PyTorch tensor.
        5. Normalize pixel values to the [0, 1] range.
    """
    """
    if model_args["nn_type"] == "CNN":
        transform = transforms.Compose([
            transforms.ToPILImage(), # consider using another method since frame stacking leads to >4 channels, which PILimage doesnt support.
            #transforms.Grayscale(),   
            #*( [transforms.Resize(model_args["preprocessing"]["resize"]["shape"])] if model_args["preprocessing"]["resize"]["enabled"] else [] ), #transforms.Resize((84, 84))
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.0], std=[1.0]),  
        ])
        tensor_frame = transform(frame)
    elif model_args["nn_type"] == "DNN":
        tensor_frame = torch.tensor(frame, dtype=torch.float32)
    """
    
    tensor_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) # (in_channels, h, w)
    
    return tensor_frame

def preprocess_data2(device, frame, model_args=None):
    tensor_frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).to(device) # (in_channels, h, w)
    
    return tensor_frame

def compile_args(path):
    """
    Compiles and returns the arguments required for the game, model, optimizer, and training configuration.
    
    Parameters:
        path (str): The path to the configuration or environment data. This may be used to load specific settings or configurations.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    game_args = config["game"]
    optimizer_args = config["optimizer"]
    model_args = config["model"]
    training_args = config["training"]
    preprocess_args = config.get("preprocessing", {})

    return game_args, model_args, optimizer_args, training_args, preprocess_args

