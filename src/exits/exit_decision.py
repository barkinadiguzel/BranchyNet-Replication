class ExitDecision:
    def __init__(self, threshold):
        self.threshold = threshold

    def should_exit(self, entropy_value):
        return entropy_value < self.threshold
