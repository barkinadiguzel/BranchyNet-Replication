import torch.nn as nn

def get_normalization(norm_type, num_features):
    norm_type = norm_type.lower()

    if norm_type == "batchnorm":
        return nn.BatchNorm2d(num_features)

    elif norm_type == "layernorm":
        return nn.LayerNorm(num_features)

    else:
        raise ValueError(f"Unsupported normalization: {norm_type}")
