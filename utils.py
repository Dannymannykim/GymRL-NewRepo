import torch
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

    Args:
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
    """
    Compute the output shape of a convolutional/pooling layer given the input shape, kernel size, stride, and padding.

    Returns:
        tuple: A tuple (output_height, output_width) representing the height and width of the output feature map.
              The values are integers that correspond to the size of the feature map after applying the convolution.
    """

    shape = (
        math.floor(((in_shape[0] + 2*padding - kernel_size) / stride) + 1), 
        math.floor(((in_shape[1] + 2*padding - kernel_size) / stride) + 1)
    )
    #print(shape)
    return shape

def preprocess_data(input, model_args):
    """
    Preprocesses an input.

    Args:
        For CNN -> frame: frame of shape (H, W, C).
        For DNN -> features: features of shape (num_features,) 

    Returns:
        For CNN -> torch.Tensor: Preprocessed frame of shape (C, H, W).
        For DNN -> torch.Tensor: Preprocessed input of shape (num_features,).

    Note: We use gym env, so certain preprocessing steps are 
    done directly within the env, including resizing and grey-scale.
    For our implementation, we'll only need to convert to tensor and
    reshape. Also, normalization is done with the nn model itself.
    """

    if model_args['nn_type'] == 'CNN':
        tensor_input = torch.tensor(input, dtype=torch.float32).permute(2, 0, 1)
    elif model_args['nn_type'] == 'DNN':
        tensor_input = torch.tensor(input, dtype=torch.float32)
    else:
        raise NotImplementedError
    return tensor_input

def compile_args(path):
    """
    Compiles and returns the arguments required for the game, model, optimizer, and training configuration.
    
    Args:
        path (str): The path to the configuration or environment data. This may be used to load specific settings or configurations.

    Returns:
        dict: game_args, model_args, optimizer_args, training_args, preprocess_args
    """

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    game_args = config["game"]
    optimizer_args = config["optimizer"]
    model_args = config["model"]
    training_args = config["training"]
    preprocess_args = config.get("preprocessing", {})
    alg_args = config["algorithm"]

    return game_args, model_args, optimizer_args, training_args, preprocess_args, alg_args


