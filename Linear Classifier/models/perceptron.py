"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        # ### Step 1: Data preprocessing
        X_use = X_train

        if self.n_class == 2:
            # ## Step 1: if labels are {0, 1}, change to {-1, 1}
            N = np.shape(X_use)[0]
            D = np.shape(X_use)[1]
            y_use = np.zeros(N)
            y_use[y_train == 0] = -1
            y_use[y_train == 1] = 1

            ### Step 2: initialize w
            if type(self.w) == type(None):
                np.random.seed(100)
                self.w = np.random.random(D+1)

            ### Step 3: update w
            for epoch in range(self.epochs):

                ### shuffle training set
                permutation = np.random.permutation(X_train.shape[0])
                X_use_new = X_use[permutation]
                y_use_new = y_use[permutation]

                for i in range(N):  # every xi
                    xi = np.append(1, X_use_new[i])
                    if y_use_new[i] * (self.w @ xi) < 0:         # different sign
                        self.w += self.lr * y_use_new[i] * xi
                ## get epoch accuracy
                pred_percept = self.predict(X_train)
                accuracy = self.get_acc(pred_percept, y_train)
                print("The training accuracy of epoch {} is {}".format(epoch, accuracy))

                ## learning rate decay
                self.lr *= 0.95
        else:
            ### Step 2: initialize w
            N = np.shape(X_use)[0]
            D = np.shape(X_use)[1]
            if type(self.w) == type(None):
                np.random.seed(100)
                self.w = np.random.random((D+1, self.n_class))

            ### Step 3: update w
            for epoch in range(self.epochs):

                ### shuffle training set
                permutation = np.random.permutation(X_train.shape[0])
                X_use_new = X_use[permutation]
                y_train_new = y_train[permutation]

                for i in range(N):          # every xi
                    xi = np.append(1, X_use_new[i])
                    for c in range(self.n_class):           # every c
                        if c != y_train_new[i]:
                            wc_x = xi @ self.w[:, c]
                            wy_x = xi @ self.w[:, int(y_train_new[i])]
                            if wc_x > wy_x:
                                self.w[:, int(y_train_new[i])] += self.lr * xi
                                self.w[:, c] -= self.lr * xi
                ## get epoch accuracy
                pred_percept = self.predict(X_train)
                accuracy = self.get_acc(pred_percept, y_train)
                print("The training accuracy is {}".format(accuracy))

                ## learning rate decay
                self.lr *= 0.99

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

        # # data preprocessing
        X_use = X_test

        # include bias
        N = np.shape(X_use)[0]
        one = np.ones(N, dtype='float64').reshape([-1, 1])
        X_test_new = np.append(one, X_use, axis=1)

        if self.n_class == 2:
            WX = X_test_new @ self.w.reshape([-1, 1])
            WX[WX >= 0] = 1
            WX[WX < 0] = 0
            result = WX[:, 0]
        else:
            WX = X_test_new @ self.w
            result = np.argmax(WX, axis=1)

        return result

    def get_acc(self, pred: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.sum(y_test == pred) / len(y_test) * 100)
