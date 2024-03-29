import numpy as np
from matplotlib import pyplot as plt
from utils import ones_for_bias_trick

# TODO add regularisation term
class LeastSquaresRegressor():
    """
    Least squares regression - i.e. maximum likelihood estimate of the weights, assuming
    Gaussian noise (i.e. Gaussian likelihood function).
    Also assuming that data matrix X is full column rank (tall).
    """
    def __init__(self, basis_function, *basis_function_args):
        self.basis_function = basis_function
        self.basis_function_args = basis_function_args

    def fit(self, X, y, estimate_var=True, visualise=False):
        """
        :param X: input data matrix - columns = features, rows = individual inputs
        :param y: labelled regression targets
        :param estimate_var: bool flag
        - estimate variance of Gaussian noise using maximum likelihood?
        :return: mean squared error, estimate of Gaussian noise variance
        """
        X_orig = X
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)
            print('X', np.amax(X), np.amin(X))

        X = ones_for_bias_trick(X)
        self.weights = (np.linalg.inv(np.matmul(X.T, X))) @ (np.dot(X.T, y))
        print('cond', np.linalg.cond(np.matmul(X.T, X)))
        mse = np.linalg.norm((y - np.dot(X, self.weights)))

        if visualise:
            self.visualise_line(X_orig, y)

        if estimate_var:
            var_estimate = 1.0/X.shape[0] * mse**2
            return mse, var_estimate
        else:
            return mse

    def predict(self, x):
        """
        :param x: test input vector - only takes a single test sample at a time
        #TODO might be a better idea to have this take a test input matrix X, with multiple
        samples?
        :return: predicted output
        """
        if self.basis_function is not None:
            x = self.basis_function(np.array([x]), *self.basis_function_args)
            x = ones_for_bias_trick(x)
            x = np.squeeze(x)
        else:
            x = np.concatenate([[1], [x]])

        return np.dot(x, self.weights)

    def visualise_line(self, X, y):
        print("Note: can only visualise 1D inputs currently!")
        axes = plt.gca()
        plt.scatter(X[:, -1], y)
        low, high = X[:, -1].min(), X[:, -1].max()
        x_vals = np.linspace(low, high, 50)
        y_vals = list(map(self.predict, x_vals))
        if self.basis_function is None:
            plt.text(0.75, 0.1, "Intercept: {} \nSlope:{}".format(round(self.weights[0], 2),
                                                                  round(self.weights[1], 2)),
                     transform=axes.transAxes)

        plt.plot(x_vals, y_vals, color='r')
        plt.xlabel("x")
        plt.ylabel("y")
        axes.legend(['Least squares fit'])
        plt.show()
