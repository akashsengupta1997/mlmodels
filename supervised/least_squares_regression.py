import numpy as np
from matplotlib import pyplot as plt


class LeastSquaresRegressor():
    """
    Least squares regression - i.e. maximum likelihood estimate of the weights, assuming
    Gaussian noise.
    Also assuming that data matrix X is full column rank (tall).
    """
    def __init__(self, input_dims, basis_function=None):
        self.input_dims = input_dims
        self.basis_function = basis_function

    def ones_for_bias_trick(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def fit(self, X, y, estimate_var=True, visualise=False):
        """
        :param X: input data matrix - columns = features, rows = individual inputs
        :param y: labelled regression targets
        :param estimate_var: bool flag
        - estimate variance of Gaussian noise using maximum likelihood?
        :return: mean squared error, estimate of Gaussian noise variance
        """

        if self.basis_function is not None:
            X = self.basis_function(X)

        X = self.ones_for_bias_trick(X)
        self.weights = (np.linalg.inv(np.matmul(X.T, X))) @ (np.dot(X.T, y))
        mse = np.linalg.norm((y - np.dot(X, self.weights)))

        if visualise:
            self.visualise_line(X, y)

        if estimate_var:
            var_estimate = 1.0/X.shape[0] * mse**2
            return mse, var_estimate
        else:
            return mse

    def predict(self, x):
        """
        :param x: test input vector
        :return: predicted output
        """
        x = np.concatenate([[1], x])
        return np.dot(x, self.weights)

    def visualise_line(self, X, y):
        assert len(self.weights) == 2, "Can only visualise 1D inputs currently :("
        axes = plt.gca()
        plt.scatter(X[:, -1], y)
        x_vals = np.array(axes.get_xlim())
        y_vals = self.weights[0] + self.weights[1] * x_vals
        plt.plot(x_vals, y_vals, '--', color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.text(0.75, 0.1, "Intercept: {} \nSlope:{}".format(round(self.weights[0], 2),
                                                         round(self.weights[1], 2)),
                 transform=axes.transAxes)
        axes.legend(['Least squares fit'])
        plt.show()
