"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me

        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        ### Step 1: initialize w
        N = np.shape(X_train)[0]
        D = np.shape(X_train)[1]
        y_use = np.zeros(N)
        y_use[y_train == 0] = -1
        y_use[y_train == 1] = 1
        if type(self.w) == type(None):
            np.random.seed(100)
            self.w = np.random.random(D)

        for epoch in range(self.epochs):

            ### Step 3: update w
            for i in range(N):
                z = -y_use[i] * (X_train[i] @ self.w)
                sigm = self.sigmoid(z)
                self.w += self.lr * sigm * y_use[i] * X_train[i]

            ## get epoch accuracy
            pred_percept = self.predict(X_train)
            accuracy = self.get_acc(pred_percept, y_train)
            print("The training accuracy is {}".format(accuracy))

            ## learning rate decay
            self.lr *= 0.95

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N = np.shape(X_test)[0]

        WX = self.sigmoid(X_test @ self.w)
        result = np.zeros(N)
        result[WX >= self.threshold] = 1

        return result

    def get_acc(self, pred: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.sum(y_test == pred) / len(y_test) * 100)
