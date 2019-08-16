import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import ones_for_bias_trick
import math


class LogisticRegressor():
    def __init__(self, basis_function, *basis_function_args):
        self.basis_function = basis_function
        self.basis_function_args= basis_function_args

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def compute_gradient(self, X, y, weights):
        """
        Computes gradient of log-likelihood w.r.t logistic regression weights
        :param X: Input data array with shape (num samples, X dims, 1).
        :param y: Target array with shape (num samples,).
        :param weights: Logistic regression weights array with shape (X dims,)
        :return: gradient of LL.
        """
        sigmoid = self.sigmoid(np.dot(X, weights))
        return np.dot(X.T, y - sigmoid)

    def compute_log_likelihood(self, X, y, weights, avg=False):
        """
        Computes log-likelihood of target labels given inputs X and weights.
        :param X: Input data array with shape (num samples, X dims, 1).
        :param y: Target array with shape (num samples,).
        :param weights: Logistic regression weights array with shape (X dims,)
        :param avg: bool flag - average log likelihood (instead of sum)?
        :return: log-likelihood.
        """
        Z = self.sigmoid(np.dot(X, weights))
        epsilon = np.finfo(float).eps
        Z = np.clip(Z, epsilon, 1.0-epsilon)

        ll_all = y * np.log(Z) + (1 - y) * np.log(1 - Z)
        if not avg:
            return np.sum(ll_all)
        else:
            return np.mean(ll_all)

    def create_batches(self, X, y, batch_size, step):
        """
        Create sequential batches of data with size = batch_size.
        :param X: Input data array with shape (num_samples, input_dims, 1). Should be SHUFFLED
        to prevent ordering of sequential batches from being an issue.
        :param y: Target array with shape (num_samples,).
        Should be SHUFFLED to prevent ordering of sequential batches from being an issue.
        :param batch_size: number of samples in one minibatch
        :param step: current step number in current epoch.
        :return: Batch of training samples and targets.
        """
        index_start = batch_size * step
        index_end = index_start + batch_size

        if index_end < X.shape[0]:
            batch_indices = np.arange(index_start, index_end)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
        else:
            batch_indices = np.arange(index_start, X.shape[0])
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

        return X_batch, y_batch

    def fit(self, X_train, y_train, lr, epochs, batch_size, X_val, y_val, optimiser='sga',
            visualise_training=False, avg_ll=False):
        """
        Update logistic regression weights using gradient-based optimisation algorithm.
        :param X_train: Training input data array with shape (num_samples, input_dims, 1).
        :param y_train: Training target array with shape (num_samples,).
        :param lr: Learning rate for gradient ascent.
        :param epochs: Number of epochs
        :param batch_size: Minibatch size (for batch logistic regression).
        :param X_val: Validation input data array with shape (num_val_samples, input_dims, 1).
        :param y_val: Validation target array with shape (num_val_samples,).
        :param optimiser: Optimisation algorithm to be used - can either be stochastic gradient
        ascent or newtons method #TODO implement newtons method.
        :param visualise_training: bool flag - visualise ll and accuracy vs epoch curves.
        """
        assert optimiser in ['sga', 'newton'], "Invalid optimiser!"

        X_orig_train = X_train
        X_orig_val = X_val

        if self.basis_function is not None:
            X_train = self.basis_function(X_train, *self.basis_function_args)
            X_val = self.basis_function(X_val, *self.basis_function_args)

        X_train = ones_for_bias_trick(X_train)
        X_val = ones_for_bias_trick(X_val)

        weights = np.random.randn(X_train.shape[1]) * 0.5
        train_log_likelihoods = []
        test_log_likelihoods = []
        train_accs = []
        test_accs = []
        steps_per_epoch = math.ceil(X_train.shape[0]/batch_size)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            for step in range(steps_per_epoch):
                X_batch, y_batch = self.create_batches(X_train, y_train, batch_size, step)
                gradient = self.compute_gradient(X_batch, y_batch, weights)
                weights = weights + lr * gradient

            self.weights = weights

            if visualise_training:
                train_log_likelihoods.append(self.compute_log_likelihood(X_train,
                                                                         y_train,
                                                                         weights,
                                                                         avg=avg_ll))
                test_log_likelihoods.append(self.compute_log_likelihood(X_val,
                                                                        y_val,
                                                                        weights,
                                                                        avg=avg_ll))
                train_accs.append(self.compute_accuracy(X_orig_train, y_train))
                test_accs.append(self.compute_accuracy(X_orig_val, y_val))

        if visualise_training:
            plt.figure(1)
            plt.plot(np.arange(1, epochs+1), train_log_likelihoods, label='Training')
            plt.plot(np.arange(1, epochs+1), test_log_likelihoods, label='Test')
            plt.legend()
            plt.show()

            plt.figure(2)
            plt.plot(np.arange(1, epochs+1), train_accs, label='Training')
            plt.plot(np.arange(1, epochs + 1), test_accs, label='Test')
            plt.legend()
            plt.show()

    def compute_accuracy(self, X_orig, y_target):
        """
        Compute classification accuracy.
        :param X_orig: Input data array (without bias trick ones).
        :param y_target: Target labels array.
        :return: accuracy.
        """
        y_output = self.predict(X_orig)
        y_output = np.around(y_output)
        matches = np.sum(y_output == y_target)
        accuracy = matches/float(y_output.shape[0])
        return accuracy

    def predict(self, X):
        """
        :param X: test input matrix - rows = test inputs, columns = features
        :return: probability of each input in X belonging to class 1.
        """
        if self.basis_function is not None:
            X = self.basis_function(X, *self.basis_function_args)

        X = ones_for_bias_trick(X)
        return self.sigmoid(np.dot(X, self.weights))

    def predict_on_grid(self, xx, yy):
        X_visualise = np.stack([xx, yy], axis=-1)
        X_visualise = np.reshape(X_visualise,
                                 (X_visualise.shape[0] * X_visualise.shape[1], -1))
        Z_visualise = self.predict(X_visualise)
        Z_visualise = Z_visualise.reshape(xx.shape)
        return Z_visualise

    def visualise_2d_input_scatter(self, X, y, show=False):
        plt.figure()
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Class 1')
        if show:
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.show()

    def visualise_2d_input_contour(self, X, y):
        self.visualise_2d_input_scatter(X, y, show=False)
        axes = plt.gca()
        x_min, x_max = 1.2 * np.array(axes.get_xlim())
        y_min, y_max = 1.2 * np.array(axes.get_ylim())
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z_visualise = self.predict_on_grid(xx, yy)
        cs = plt.contour(xx, yy, Z_visualise, cmap='RdBu', linewidths=1, levels=10)
        plt.clabel(cs, fmt='%2.1f', colors='k', fontsize=14)
        axes.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def visualise_2d_input_surface(self, X, y):
        fig = plt.figure()
        axes = Axes3D(fig)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z_visualise = self.predict_on_grid(xx, yy)
        surf = axes.plot_surface(xx, yy, Z_visualise,cmap='RdBu',
                                 linewidth=0, antialiased=False)
        axes.scatter(X[y == 0, 0], X[y == 0, 1], zs=0, c='r', label='Class 0')
        axes.scatter(X[y == 1, 0], X[y == 1, 1], zs=0, c='b', label='Class 1')
        plt.show()





