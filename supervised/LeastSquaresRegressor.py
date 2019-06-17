import numpy as np


class LeastSquaresRegressor():
    """
    Least squares regression - i.e. maximum likelihood regression assuming Gaussian noise.
    Also assuming that data matrix X is full column rank (tall).
    """
    def __init__(self, input_dims, basis_function=None):
        self.input_dims = input_dims
        self.basis_function = basis_function
        self.weights = np.zeros(input_dims)

    def ones_for_bias_trick(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y):

        if self.basis_function is not None:
            X = self.basis_function(X)

        X = self.ones_for_bias_trick(X)
        self.weights = (np.linalg.inv(np.matmul(X.T, X))) @ (np.dot(X.T, y))

    def predict(self, x):
        x = np.concatenate([1, x])
        return np.dot(x, self.weights)
