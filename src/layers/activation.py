import torch.nn as nn

def get_activation(name="relu", negative_slope=0.01):
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    elif name == "sigmoid":
        return nn.Sigmoid()

    elif name == "tanh":
        return nn.Tanh()

    else:
        raise ValueError(f"Unsupported activation: {name}")
