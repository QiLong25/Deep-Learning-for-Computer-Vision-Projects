"""Neural network model."""

from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size:int):
        """
        Initialize neural network.
        """
        super().__init__()

        self.input_size = input_size

        self.sequence = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.ReLU(),
        )

    def forward(self, X):
        """
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C)
        """

        y = self.sequence(X)
        return y
