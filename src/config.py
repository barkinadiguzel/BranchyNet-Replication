NUM_CLASSES = 10          # number of classes (e.g. MNIST/CIFAR10)
INPUT_CHANNELS = 3        # input image channels (RGB)

# Backbone blocks: (in_channels, out_channels)
BACKBONE_CONFIG = [
    (INPUT_CHANNELS, 64),
    (64, 128),
    (128, 256)
]

# Branch blocks: in/out channels + early exit threshold
BRANCH_CONFIGS = [
    {'in_channels': 64, 'out_channels': 64, 'threshold': 0.3},   # first branch
    {'in_channels': 128, 'out_channels': 128, 'threshold': 0.25} # second branch
]

# Weights for joint loss (optional)
JOINT_LOSS_WEIGHTS = {
    'exit1': 0.3,
    'exit2': 0.3,
    'final': 0.4
}
