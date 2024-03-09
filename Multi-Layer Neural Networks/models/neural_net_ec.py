"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        ### Parameter initialization
        self.params = {}            # {"W1" : np.array, "b1" : np.array, ... }
        self.m_adam = {}            # {"W1" : np.array, "b1" : np.array, ... }
        self.v_adam = {}            # {"W1" : np.array, "b1" : np.array, ... }
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # TODO: You may set parameters for Adam optimizer here
            self.opt = opt
            if self.opt == "Adam":
                self.m_adam["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.m_adam["b" + str(i)] = np.zeros(sizes[i])
                self.v_adam["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
                self.v_adam["b" + str(i)] = np.zeros(sizes[i])
                self.lamb_adam = 0.0001
                self.epoch = 0

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me

        de_dw = X.T @ de_dz
        de_db = np.ones(np.shape(de_dz)[0]).reshape([1, -1]) @ de_dz
        de_dx = de_dz @ W.T

        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return X * (X > 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return 1 * (X > 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.sum((y - p) ** 2) / np.shape(y)[0]

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return -2 * (y - p) / np.shape(y)[0]

    def l1loss(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sum(np.absolute(y - p)) / np.shape(y)[0]

    def l1loss_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (-1 * ((y - p) >= 0) + 1 * ((y - p) < 0)) / np.shape(y)[0]
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return self.mse_grad(y, p) * self.sigmoid_grad(p)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs["x"] = X

        for layer_idx in range(1, self.num_layers + 1):

            ## Linear layers
            if layer_idx == 1:
                self.outputs["z" + str(layer_idx)] = self.linear(self.params["W" + str(layer_idx)],
                                                                 X,
                                                                 self.params["b" + str(layer_idx)])
            else:
                self.outputs["z" + str(layer_idx)] = self.linear(self.params["W" + str(layer_idx)],
                                                                 self.outputs["h" + str(layer_idx - 1)],
                                                                 self.params["b" + str(layer_idx)])

            ## Non-linear layers
            self.outputs["h" + str(layer_idx)] = self.relu(self.outputs["z" + str(layer_idx)])

        if self.opt == "Adam":
            self.epoch += 1

        return self.outputs["h" + str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        ## compute loss e
        loss = float(self.mse(y, self.outputs["h" + str(self.num_layers)]))
        self.gradients["de_dh" + str(self.num_layers)] = self.mse_grad(y, self.outputs["h" + str(self.num_layers)])

        ## l1 loss
        # loss = float(self.l1loss(y, self.outputs["h" + str(self.num_layers)]))
        # self.gradients["de_dh" + str(self.num_layers)] = self.l1loss_grad(y, self.outputs["h" + str(self.num_layers)])

        for i in range(1, self.num_layers + 1):

            ## start from last layer
            layer_idx = self.num_layers + 1 - i

            ## de_dz = de_dh * dh_dz
            self.gradients["de_dz" + str(layer_idx)] = self.gradients["de_dh" + str(layer_idx)] * self.relu_grad(self.outputs["z" + str(layer_idx)])

            ## de_dW = de_dz * dz_dW
            ## de_db = de_dz * dz_db
            ## de_dh(t-1) = de_dz * dz_dh(t-1)
            if layer_idx == 1:
                de_dw, de_db, de_dx = self.linear_grad(self.params["W" + str(layer_idx)],
                                                       self.outputs["x"],
                                                       self.params["b" + str(layer_idx)],
                                                       self.gradients["de_dz" + str(layer_idx)], None, None)
                self.gradients["W" + str(layer_idx)] = de_dw
                self.gradients["b" + str(layer_idx)] = de_db
            else:
                de_dw, de_db, de_dx = self.linear_grad(self.params["W" + str(layer_idx)],
                                                       self.outputs["h" + str(layer_idx - 1)],
                                                       self.params["b" + str(layer_idx)],
                                                       self.gradients["de_dz" + str(layer_idx)], None, None)
                self.gradients["W" + str(layer_idx)] = de_dw
                self.gradients["b" + str(layer_idx)] = de_db
                self.gradients["de_dh" + str(layer_idx - 1)] = de_dx

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        ### SGD
        if opt == "SGD":
            for layer_idx in range(1, self.num_layers + 1):
                self.params["W" + str(layer_idx)] -= lr * self.gradients["W" + str(layer_idx)]
                self.params["b" + str(layer_idx)] -= lr * self.gradients["b" + str(layer_idx)].reshape([-1])

        if opt == "Adam":
            for layer_idx in range(1, self.num_layers + 1):
                self.gradients["b" + str(layer_idx)] = self.gradients["b" + str(layer_idx)].reshape([-1])

                ## regularization
                if self.lamb_adam != 0:
                    self.gradients["W" + str(layer_idx)] += self.lamb_adam * self.params["W" + str(layer_idx)]
                    self.gradients["b" + str(layer_idx)] += self.lamb_adam * self.params["b" + str(layer_idx)]

                ## update m, v
                self.m_adam["W" + str(layer_idx)] = b1 * self.m_adam["W" + str(layer_idx)]\
                                                    + (1 - b1) * self.gradients["W" + str(layer_idx)]
                self.m_adam["b" + str(layer_idx)] = b1 * self.m_adam["b" + str(layer_idx)]\
                                                    + (1 - b1) * self.gradients["b" + str(layer_idx)]
                self.v_adam["W" + str(layer_idx)] = b2 * self.v_adam["W" + str(layer_idx)] \
                                                    + (1 - b2) * (self.gradients["W" + str(layer_idx)] ** 2)
                self.v_adam["b" + str(layer_idx)] = b2 * self.v_adam["b" + str(layer_idx)] \
                                                    + (1 - b2) * (self.gradients["b" + str(layer_idx)] ** 2)

                ## compute m_hat, v_hat
                m_hat_W = self.m_adam["W" + str(layer_idx)] / (1 - b1 ** self.epoch)
                m_hat_b = self.m_adam["b" + str(layer_idx)] / (1 - b1 ** self.epoch)
                v_hat_W = self.v_adam["W" + str(layer_idx)] / (1 - b2 ** self.epoch)
                v_hat_b = self.v_adam["b" + str(layer_idx)] / (1 - b2 ** self.epoch)

                ## update parameters
                self.params["W" + str(layer_idx)] -= lr * m_hat_W / (v_hat_W ** 0.5 + eps)
                self.params["b" + str(layer_idx)] -= lr * m_hat_b / (v_hat_b ** 0.5 + eps)

        return