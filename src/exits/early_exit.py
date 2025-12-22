import torch.nn as nn
import torch.nn.functional as F

class EarlyExit(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
