"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me

        GW = np.zeros(np.shape(self.w))
        N = y_train.shape[0]

        ### Regularization term
        GW += self.reg_const * self.w

        Pred = X_train @ self.w

        if self.n_class == 2:
            y_use = np.zeros(N)
            y_use[y_train == 0] = -1
            y_use[y_train == 1] = 1

            Pred = Pred.reshape([-1, 1])
            Pred = np.append(Pred, -Pred, axis=1)
            pexp = self.softmax(Pred)
            pexp[range(N), y_train] -= 1
            pexp = pexp / N
            y_hat = np.argmax(Pred, axis=1)
            for i in range(N):
                GW += pexp[i][y_hat[i]] * X_train[i] * y_use[i]

        else:
            pexp = self.softmax(Pred)
            pexp[range(N), y_train] -= 1
            pexp = pexp / N
            GW += X_train.T @ pexp

        return GW

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me

        ### SGD
        batch_size = 1500

        ### Step 1: Data preprocessing
        X_use = X_train

        if self.n_class == 2:
            print("Doing binary classification.")

            X_mean = np.mean(X_train, axis=0)
            X_use = X_train - X_mean
            X_use_max = np.max(np.absolute(X_use), axis=0)
            X_use = X_use / X_use_max

            ### Step 1: initialize w
            N = np.shape(X_use)[0]
            D = np.shape(X_use)[1]
            if type(self.w) == type(None):
                np.random.seed(100)
                self.w = np.random.random(D)

        else:
            ### Step 1: initialize w
            N = np.shape(X_use)[0]
            D = np.shape(X_use)[1]
            if type(self.w) == type(None):
                np.random.seed(100)
                self.w = np.random.random((D, self.n_class))

        for epoch in range(self.epochs):
            ### Step 2: fetch mini-batch
            row_idxs = np.arange(N)
            np.random.shuffle(row_idxs)
            X_mini = X_use[row_idxs[0:batch_size], :]
            y_mini = y_train[row_idxs[0:batch_size]]

            ### Step 3: update w
            GW = self.calc_gradient(X_mini, y_mini)
            self.w -= self.lr * GW

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

        # data preprocessing
        X_use = X_test

        if self.n_class == 2:
            X_mean = np.mean(X_test, axis=0)
            X_use = X_test - X_mean
            X_use_max = np.max(np.absolute(X_use), axis=0)
            X_use = X_use / X_use_max

        WX = X_use @ self.w
        if self.n_class == 2:
            WX = WX.reshape([-1, 1])
            WX = np.append(WX, -WX, axis=1)
        result = np.argmax(WX, axis=1)

        return result

    def softmax(self, pred: np.ndarray) -> np.ndarray:
        ## pred should be N * C
        exps = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def get_acc(self, pred: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.sum(y_test == pred) / len(y_test) * 100)
