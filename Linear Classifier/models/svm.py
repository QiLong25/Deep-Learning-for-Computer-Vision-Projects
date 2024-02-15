"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me

        if self.n_class == 2:
            print("Doing binary classification.")
            ### Step 1: if labels are {0, 1}, change to {-1, 1}
            N = np.shape(X_train)[0]
            D = np.shape(X_train)[1]
            y_use = np.zeros(N)
            y_use[y_train == 0] = -1
            y_use[y_train == 1] = 1
            GW = np.zeros(np.shape(self.w))

            ### Step 2: calculate first term
            GW += self.reg_const * self.w

            ### Step 3: calculate second term
            for i in range(N):  # every xi
                xi = X_train[i]
                if y_use[i] * xi @ self.w < 1:
                    GW -= y_use[i] * xi / N

        else:

            N = np.shape(X_train)[0]
            D = np.shape(X_train)[1]
            GW = np.zeros(np.shape(self.w))

            ### Step 2: calculate first term
            for i in range(N):          # every xi
                for c in range(self.n_class):           # every c
                    GW[:, c] += self.reg_const / N * self.w[:, c]

            ### Step 3: calculate second term
            for i in range(N):          # every xi
                xi = X_train[i]
                for c in range(self.n_class):           # every c
                    wc_x = xi @ self.w[:, c]
                    wy_x = xi @ self.w[:, int(y_train[i])]
                    if wy_x - wc_x < 1 and c != y_train[i]:
                        GW[:, int(y_train[i])] -= xi / N
                        GW[:, c] += xi / N

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
        batch_size = 1024

        X_use = X_train

        if self.n_class == 2:
            ### Step 1: Data preprocessing
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

        X_use = X_test

        if self.n_class == 2:
            # data preprocessing
            X_mean = np.mean(X_test, axis=0)
            X_use = X_test - X_mean
            X_use_max = np.max(np.absolute(X_use), axis=0)
            X_use = X_use / X_use_max

            WX = X_use @ self.w.reshape([-1, 1])
            WX[WX >= 0] = 1
            WX[WX < 0] = 0
            result = WX[:, 0]
        else:
            WX = X_use @ self.w
            result = np.argmax(WX, axis=1)

        return result

    def get_acc(self, pred: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.sum(y_test == pred) / len(y_test) * 100)
