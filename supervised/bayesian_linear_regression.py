import numpy as np


class BayesianLinearRegressor():
    def __init__(self, prior_precision, noise_var):
        self.prior_precision = prior_precision
        self.noise_var = noise_var

    def compute_posterior(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """

        self.posterior_cov = np.linalg.inv((1.0/self.noise_var) * np.matmul(X.T, X)
                                      + self.prior_precision * np.eye(X.shape[1]))

        self.posterior_mean = (1.0/self.noise_var) * np.matmul(self.posterior_cov, np.dot(X.T, y))

    def compute_predictive_distribution(self, x):
        """

        :param x:
        :return:
        """

        predictive_mean = np.dot(self.posterior_mean, x)
        predictive_cov = np.dot(x.T, np.dot(self.posterior_cov, x)) + self.noise_var

        return predictive_mean, predictive_cov