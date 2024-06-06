class Optimizer:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, iterations):
        super().__init__(learning_rate, iterations)

    def update(self, weights, bias, weights_slope ,bias_slope):
        weights = weights - self.learning_rate * weights_slope
        bias = bias - self.learning_rate * bias_slope
        return weights, bias

