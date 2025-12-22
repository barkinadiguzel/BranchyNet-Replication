import torch
import torch.nn as nn

def joint_loss(exit_logits, target, weights=None):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    if weights is None:
        # default equal weights
        num_exits = len(exit_logits)
        weights = {k: 1.0 / num_exits for k in exit_logits.keys()}

    for exit_name, logits in exit_logits.items():
        w = weights.get(exit_name, 1.0)
        loss = loss_fn(logits, target)
        total_loss += w * loss

    return total_loss
