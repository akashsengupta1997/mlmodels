import numpy as np
from matplotlib import pyplot as plt


class BayesianLinearRegressor():
    """
    Bayesian linear regressor assuming Gaussian prior over weights and Gaussian noise (i.e.
    Gaussian likelihood) - allowing for exact inference.
    """
    def __init__(self, prior_precision, noise_var):
        self.prior_precision = prior_precision
        self.noise_var = noise_var

    def ones_for_bias_trick(self, X):
        return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    def compute_posterior(self, X, y, visualise=False):
        """

        :param X:
        :param y:
        :return:
        """
        X = self.ones_for_bias_trick(X)
        self.posterior_cov = np.linalg.inv((1.0/self.noise_var) * np.matmul(X.T, X)
                                           + self.prior_precision * np.eye(X.shape[1]))

        self.posterior_mean = (1.0/self.noise_var) * np.matmul(self.posterior_cov,
                                                               np.dot(X.T, y))

        if visualise:
            self.visualise_line(X, y)

    def compute_predictive_distribution(self, x):
        """

        :param x:
        :return:
        """
        x = np.concatenate([[1], [x]])
        predictive_mean = np.dot(self.posterior_mean, x)
        predictive_cov = np.dot(x.T, np.dot(self.posterior_cov, x)) + self.noise_var

        return predictive_mean, predictive_cov

    def visualise_line(self, X, y):
        assert len(self.posterior_mean) == 2, "Can only visualise 1D inputs currently :("
        axes = plt.gca()
        plt.scatter(X[:, -1], y)
        low, high = axes.get_xlim()
        x_vals = np.linspace(low, high, 50)
        mean_vals, var_vals = zip(*list(map(self.compute_predictive_distribution, x_vals)))
        mean_vals = np.array(mean_vals)
        var_vals = np.array(var_vals)
        one_std_above_vals = mean_vals + np.sqrt(var_vals)
        one_std_below_vals = mean_vals - np.sqrt(var_vals)
        plt.plot(x_vals, mean_vals, '-', color='r')
        plt.plot(x_vals, one_std_above_vals, '--', color='gray')
        plt.plot(x_vals, one_std_below_vals, '--', color='gray')
        plt.show()
