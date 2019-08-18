import numpy as np
from matplotlib import pyplot as plt
from utils import ones_for_bias_trick


class BayesianLinearRegressor():
    """
    Bayesian linear regressor assuming Gaussian prior over weights and Gaussian noise (i.e.
    Gaussian likelihood) - allowing for exact inference.
    """
    def __init__(self, prior_precision, noise_var, basis_function, *basis_function_args):
        """
        :param prior_precision: reciprocal of Gaussian prior variance
        :param noise_var: Gaussian noise variance (for likelihood)
        """
        self.prior_precision = prior_precision
        self.noise_var = noise_var
        self.basis_function = basis_function
        self.basis_function_args = basis_function_args

    def compute_posterior(self, X, y, visualise=False):
        """
        Compute posterior probability of weights given training data, i.e. p(w|X, y)
        :param X:
        :param y:
        :return:
        """
        X_orig = X
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)

        X = ones_for_bias_trick(X)
        self.posterior_cov = np.linalg.inv((1.0/self.noise_var) * np.matmul(X.T, X)
                                           + self.prior_precision * np.eye(X.shape[1]))

        self.posterior_mean = (1.0/self.noise_var) * np.matmul(self.posterior_cov,
                                                               np.dot(X.T, y))

        if visualise:
            self.visualise_line(X_orig, y)

    def compute_predictive_distribution(self, x):
        """
        Compute predictive distribution of output y*, given test input x*, training data X, y,
        i.e. p(y*|x*, X, y)
        :param x: test input vector.
        :return: predictive distribution mean vector and covariance matrix.
        """
        if self.basis_function is not None:
            x = self.basis_function(np.array([x]), *self.basis_function_args)
            x = ones_for_bias_trick(x)
            x = np.squeeze(x)
        else:
            x = np.concatenate([[1], [x]])

        predictive_mean = np.dot(np.squeeze(self.posterior_mean), x)
        predictive_cov = np.dot(x.T, np.dot(self.posterior_cov, x)) + self.noise_var

        return predictive_mean, predictive_cov

    def compute_log_marginal_likelihood(self, X, y):
        """
        Computes model evidence/log marginal likelihood - i.e. p(y|X)
        :param X:
        :param y:
        :return:
        """
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)
        X = ones_for_bias_trick(X)
        ml_cov = (1.0/self.prior_precision) * np.matmul(X, X.T) + self.noise_var * np.eye(X.shape[0])
        log_ml = self.log_gaussian_pdf(np.zeros(X.shape[0]), ml_cov, y)
        return log_ml

    def log_gaussian_pdf(self, mean, cov, x):
        """
        Computes log of gaussian with given mean and cov at input vector x.
        :param mean:
        :param cov:
        :param x:
        :return:
        """
        _, log_denom = np.linalg.slogdet(2*np.pi*cov)
        exp_term = np.dot((x + mean).T, np.dot(np.linalg.inv(cov), (x + mean)))
        log_prob = -0.5 * (log_denom + exp_term)
        return log_prob

    def visualise_line(self, X, y):
        print("Note: can only visualise 1D inputs currently!")
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

        if self.basis_function is None:
            plt.text(0.75, 0.1, "Posterior Means: \n Intercept: {} \n Slope:{}".format(
                round(self.posterior_mean[0], 2),
                round(self.posterior_mean[1], 2)),
                     transform=axes.transAxes)
        axes.legend(['Predictive distribution mean', '+/- 1 s.d.'])
        plt.show()
