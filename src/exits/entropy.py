import torch

def entropy(probs, eps=1e-8):
    return -torch.sum(probs * torch.log(probs + eps), dim=1)
