import torch
import torch.nn as nn

from src.blocks.backbone_block import BackboneBlock
from src.blocks.branch_block import BranchBlock
from src.blocks.classifier_head import ClassifierHead
from src.exits.entropy import compute_entropy
from src.exits.exit_decision import ExitDecision

class BranchyNet(nn.Module):
    def __init__(self, num_classes, branch_configs):
        super().__init__()

        # Backbone blocks
        self.backbone1 = BackboneBlock(3, 64)
        self.backbone2 = BackboneBlock(64, 128)
        self.backbone3 = BackboneBlock(128, 256)

        # Branches
        self.branches = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.exit_decisions = []

        for cfg in branch_configs:
            branch = BranchBlock(cfg['in_channels'], cfg['out_channels'])
            classifier = ClassifierHead(cfg['out_channels'], num_classes)
            decision = ExitDecision(cfg['threshold'])

            self.branches.append(branch)
            self.classifiers.append(classifier)
            self.exit_decisions.append(decision)

        # Final classifier (last exit)
        self.final_classifier = ClassifierHead(256, num_classes)
    def forward(self, x, fast_inference=False):
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.backbone3(x)

        if not fast_inference:
            # Training: return all exit logits for joint loss
            exit_logits = []
            for branch, classifier in zip(self.branches, self.classifiers):
                feat = branch(x)
                logits, _ = classifier(feat)
                exit_logits.append(logits)
            # Final classifier
            final_logits, _ = self.final_classifier(x)
            exit_logits.append(final_logits)
            return exit_logits
        else:
            # Inference: early exit logic
            for branch, classifier, decision in zip(self.branches, self.classifiers, self.exit_decisions):
                feat = branch(x)
                logits, probs = classifier(feat)
                e = compute_entropy(probs)
                if decision.should_exit(e):
                    return logits  # early exit
            # Final classifier if no early exit
            logits, _ = self.final_classifier(x)
            return logits
