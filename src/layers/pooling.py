import torch.nn as nn

def get_pooling(pool_type="max", kernel_size=2, stride=2):
    pool_type = pool_type.lower()

    if pool_type == "max":
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    elif pool_type == "avg":
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")
