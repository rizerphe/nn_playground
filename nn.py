"""A very basic neural network implementation"""
from abc import ABC, abstractmethod
import math


class Layer(ABC):
    """A generic neural network layer"""

    @property
    @abstractmethod
    def n_neurons(self):
        """Number of neurons in the layer"""

    @abstractmethod
    def propagate(self, inputs: list[float]) -> list[float]:
        """Propagate inputs through the layer"""

    @abstractmethod
    def backpropagate(self, errors: list[float]) -> list[float]:
        """Backpropagate error through the layer"""

    @abstractmethod
    def reset_grad(self):
        """Reset the layer's accumulated gradients"""

    @abstractmethod
    def update(self, learning_rate: float):
        """Update the layer's weights"""


class Linear(Layer):
    """A linear layer"""

    def __init__(self, weights: list[list[float]], bias: list[float]):
        """Initialize the layer with weights and bias"""
        self.weights = weights
        self.bias = bias
        self.inputs: list[float] | None = None
        self.acc_grad = [[0.0] * len(weights) for weights in self.weights] + [
            [0.0] * len(self.bias)
        ]

    @property
    def n_neurons(self):
        """Number of neurons in the layer"""
        return len(self.weights)

    def propagate(self, inputs: list[float]) -> list[float]:
        """Propagate inputs through the layer"""
        self.inputs = inputs
        return [
            sum(weight * input_ for weight, input_ in zip(weights, inputs)) + bias
            for weights, bias in zip(self.weights, self.bias)
        ]

    def backpropagate(self, errors: list[float]) -> list[float]:
        """Backpropagate error through the layer"""
        if self.inputs is None:
            raise RuntimeError("No inputs to backpropagate")
        for i, error in enumerate(errors):
            for j, input_ in enumerate(self.inputs):
                self.acc_grad[i][j] += error * input_
        for i, error in enumerate(errors):
            self.acc_grad[-1][i] += error
        return [
            sum(weight * error for weight, error in zip(weights, errors))
            for weights in zip(*self.weights)
        ]

    def reset_grad(self):
        """Reset the layer's accumulated gradients"""
        self.acc_grad = [[0.0] * len(weights) for weights in self.weights] + [
            [0.0] * len(self.bias)
        ]

    def update(self, learning_rate: float):
        """Update the layer's weights"""
        self.weights = [
            [
                weight - learning_rate * gradient
                for weight, gradient in zip(weights, gradients)
            ]
            for weights, gradients in zip(self.weights, self.acc_grad)
        ]
        self.bias = [
            bias - learning_rate * gradient
            for bias, gradient in zip(self.bias, self.acc_grad[-1])
        ]


class Sigmoid(Layer):
    """A sigmoid layer"""

    def __init__(self, n_neurons: int):
        """Initialize the layer with number of neurons"""
        self._n_neurons = n_neurons
        self.inputs: list[float] | None = None

    @property
    def n_neurons(self):
        return self._n_neurons

    def propagate(self, inputs: list[float]) -> list[float]:
        """Propagate inputs through the layer"""
        self.inputs = inputs
        return [self.sigmoid(input_) for input_ in inputs]

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def backpropagate(self, errors: list[float]) -> list[float]:
        """Backpropagate error through the layer"""
        if self.inputs is None:
            raise RuntimeError("No inputs to backpropagate")
        return [
            error * self.sigmoid(input_) * (1 - self.sigmoid(input_))
            for input_, error in zip(self.inputs, errors)
        ]

    def reset_grad(self):
        """Reset the layer's accumulated gradients"""

    def update(self, learning_rate: float):
        """Update the layer's weights"""


class Loss(ABC):
    """A loss function"""

    @abstractmethod
    def eval(self, outputs: list[float], targets: list[float]) -> float:
        """Evaluate the loss"""

    @abstractmethod
    def backpropagate(self, outputs: list[float], targets: list[float]) -> list[float]:
        """Backpropagate error"""


class MSE(Loss):
    """Mean squared error"""

    def eval(self, outputs: list[float], targets: list[float]) -> float:
        """Evaluate the loss"""
        return sum((output - target) ** 2 for output, target in zip(outputs, targets))

    def backpropagate(self, outputs: list[float], targets: list[float]) -> list[float]:
        """Backpropagate error"""
        return [2 * (output - target) for output, target in zip(outputs, targets)]


class Network:
    """A neural network"""

    def __init__(self, layers: list[Layer], loss: Loss):
        """Initialize the network with layers"""
        self.layers = layers
        self.loss = loss

    def propagate(self, inputs: list[float]) -> list[float]:
        """Propagate inputs through the network"""
        for layer in self.layers:
            inputs = layer.propagate(inputs)
        return inputs

    def backpropagate(self, outputs: list[float], targets: list[float]):
        """Backpropagate error through the network"""
        errors = self.loss.backpropagate(outputs, targets)
        for layer in reversed(self.layers):
            errors = layer.backpropagate(errors)

    def reset_grad(self):
        """Reset the network's accumulated gradients"""
        for layer in self.layers:
            layer.reset_grad()

    def update(self, learning_rate: float):
        """Update the network's weights"""
        for layer in self.layers:
            layer.update(learning_rate)

    def train(
        self,
        inputs: list[list[float]],
        targets: list[list[float]],
        learning_rate: float,
        n_epochs: int,
    ):
        """Train the network"""
        for _ in range(n_epochs):
            for input_, target in zip(inputs, targets):
                outputs = self.propagate(input_)
                self.backpropagate(outputs, target)
            self.update(learning_rate * len(inputs))
            self.reset_grad()


xor_questions = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
xor_answers = [
    [0.0],
    [1.0],
    [1.0],
    [0.0],
]
network = Network(
    [
        Linear([[0.5, 0.5], [0.5, 0.25]], [0.0, 0.0]),
        Sigmoid(2),
        Linear([[0.5, 0.5]], [0.0]),
        Sigmoid(1),
    ],
    MSE(),
)
network.train(xor_questions, xor_answers, 0.1, 10000)
for question, answer in zip(xor_questions, xor_answers):
    print(question, network.propagate(question), answer)
